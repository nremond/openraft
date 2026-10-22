//! Liveness-phase convergence check.
//!
//! After the safety phase, the fuzzer removes every fault (partitions, holds,
//! latency, loss, crashes) and requires the cluster to *completely heal*: one
//! leader, a settled (non-joint) membership, and every member holding an
//! identical, fully-applied log and state machine. A cluster that cannot reach
//! this state without faults has a liveness bug (stuck replication, dead-end
//! progress state, election live-lock, ...).

use std::collections::BTreeSet;

use crate::cluster::FullNodeSnapshot;
use crate::typ::NodeId;

/// The healed state: the leader and the members that were verified identical.
#[derive(Debug, Clone)]
pub struct Converged {
    pub leader: NodeId,
    /// All nodes of the final membership: voters and learners.
    pub members: BTreeSet<NodeId>,
}

/// The nodes that report themselves `Leader` and belong to the surviving
/// cluster: the newest effective membership any running node holds, by log id.
///
/// A node outside it may still report itself `Leader`. Removed while it could
/// not hear of a newer leader, it is sent nothing any more, and openraft lets
/// it keep that stale local state ([#2178]). It can commit nothing, so it is
/// not a leader of the cluster. A leader that was demoted to *learner* still
/// counts: openraft keeps such a leader leading (it commits through the voter
/// quorum), and `StepDownWatcher` only steps down leaders that were removed
/// from the config entirely.
///
/// [#2178]: https://github.com/databendlabs/openraft/issues/2178
pub fn surviving_leaders(snapshots: &[(NodeId, FullNodeSnapshot)]) -> Vec<NodeId> {
    let Some(newest) = snapshots.iter().map(|(_, s)| s.raft.membership_config.as_ref()).max_by_key(|m| *m.log_id())
    else {
        return Vec::new();
    };
    let m = newest.membership();
    let members: BTreeSet<NodeId> = m.voter_ids().chain(m.non_write_voter_ids()).collect();

    snapshots
        .iter()
        .filter(|(id, s)| s.raft.state.is_leader() && members.contains(id))
        .map(|(id, _)| *id)
        .collect()
}

/// Check whether the cluster has fully converged.
///
/// Returns `Err(reason)` describing the first unmet condition; the caller
/// retries until a deadline and reports the last reason on failure.
///
/// Leaders are counted from the surviving cluster's point of view, see
/// [`surviving_leaders`]: a removed node's stale `Leader` state does not stop
/// the cluster from converging.
pub fn check_converged(snapshots: &[(NodeId, FullNodeSnapshot)]) -> Result<Converged, String> {
    let claimants: Vec<NodeId> =
        snapshots.iter().filter(|(_, s)| s.raft.state.is_leader()).map(|(id, _)| *id).collect();

    let leaders = surviving_leaders(snapshots);

    if leaders.len() != 1 {
        return Err(format!(
            "expected exactly one leader, claimants={claimants:?}, surviving-cluster leaders={leaders:?}"
        ));
    }
    let leader = leaders[0];
    let (_, ls) = snapshots.iter().find(|(id, _)| *id == leader).expect("leader must be in snapshots");

    let membership = ls.raft.membership_config.membership();
    if membership.get_joint_config().len() != 1 {
        return Err(format!("leader n{leader} still in joint membership {membership}"));
    }

    let member_ids = membership.voter_ids().chain(membership.non_write_voter_ids());
    let members: BTreeSet<NodeId> = member_ids.collect();

    // The leader itself must have applied everything it holds.
    let leader_applied = ls.raft.last_applied;
    if leader_applied.map(|l| l.index) != ls.raft.last_log_index {
        return Err(format!(
            "leader n{leader} not fully applied: applied={:?}, last_log_index={:?}",
            leader_applied, ls.raft.last_log_index
        ));
    }

    for member in &members {
        let Some((_, s)) = snapshots.iter().find(|(id, _)| id == member) else {
            return Err(format!("member n{member} is not running"));
        };
        if s.raft.last_log_index != ls.raft.last_log_index {
            return Err(format!(
                "member n{member} last_log_index={:?} != leader's {:?}",
                s.raft.last_log_index, ls.raft.last_log_index
            ));
        }
        if s.raft.last_applied != leader_applied {
            return Err(format!(
                "member n{member} applied={:?} != leader's {:?}",
                s.raft.last_applied, leader_applied
            ));
        }
        if s.sm.last_applied != s.raft.last_applied {
            return Err(format!(
                "member n{member} state machine applied={:?} lags metrics applied={:?}",
                s.sm.last_applied, s.raft.last_applied
            ));
        }
        if s.sm.data != ls.sm.data {
            return Err(format!(
                "member n{member} state machine data differs from leader n{leader}"
            ));
        }
    }

    Ok(Converged { leader, members })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openraft::ServerState;
    use openraft::metrics::RaftMetrics;
    use openraft::vote::RaftLeaderId;

    use super::*;
    use crate::store::StateMachineData;
    use crate::store::ValueMeta;
    use crate::typ::LogId;
    use crate::typ::StoredMembership;
    use crate::typ::TypeConfig;

    fn log_id(term: u64, node_id: NodeId, index: u64) -> LogId {
        LogId::new(openraft::impls::leader_id_adv::LeaderId::new(term, node_id), index)
    }

    /// A node fixture: `state`, voters/learners of its own effective config,
    /// log/applied position, and one data key to control SM equality.
    fn node(
        id: NodeId,
        state: ServerState,
        voters: Vec<Vec<NodeId>>,
        learners: Vec<NodeId>,
        applied: Option<LogId>,
        data_value: &str,
    ) -> (NodeId, FullNodeSnapshot) {
        let mut raft = RaftMetrics::<TypeConfig>::new_initial(id);
        raft.state = state;
        raft.last_log_index = applied.map(|l| l.index);
        raft.last_applied = applied;

        let sets: Vec<std::collections::BTreeSet<NodeId>> =
            voters.into_iter().map(|v| v.into_iter().collect()).collect();
        let mem = openraft::Membership::new_with_defaults(sets, learners);
        raft.membership_config = Arc::new(StoredMembership::new(Some(log_id(1, 1, 1)), mem));

        let mut sm = StateMachineData {
            last_applied: applied,
            ..Default::default()
        };
        sm.data.insert("k".to_string(), ValueMeta {
            value: data_value.to_string(),
            serial: 0,
            log_id: log_id(1, 1, 1),
        });

        (id, FullNodeSnapshot {
            raft,
            sm,
            incarnation: 1,
        })
    }

    #[test]
    fn converged_cluster_accepted() {
        let applied = Some(log_id(2, 1, 10));
        let snapshots = vec![
            node(1, ServerState::Leader, vec![vec![1, 2, 3]], vec![4], applied, "v"),
            node(2, ServerState::Follower, vec![vec![1, 2, 3]], vec![4], applied, "v"),
            node(3, ServerState::Follower, vec![vec![1, 2, 3]], vec![4], applied, "v"),
            node(4, ServerState::Learner, vec![vec![1, 2, 3]], vec![4], applied, "v"),
        ];
        let converged = check_converged(&snapshots).unwrap();
        assert_eq!(converged.leader, 1);
        assert_eq!(converged.members, [1, 2, 3, 4].into_iter().collect());
    }

    #[test]
    fn no_leader_rejected() {
        let applied = Some(log_id(2, 1, 10));
        let snapshots = vec![
            node(1, ServerState::Follower, vec![vec![1, 2]], vec![], applied, "v"),
            node(2, ServerState::Follower, vec![vec![1, 2]], vec![], applied, "v"),
        ];
        assert!(check_converged(&snapshots).is_err());
    }

    #[test]
    fn ghost_leader_outside_its_config_ignored() {
        let applied = Some(log_id(2, 1, 10));
        // Node 9 claims leadership but is not a voter of its own config:
        // a stale ghost. Node 1 is the real leader.
        let snapshots = vec![
            node(1, ServerState::Leader, vec![vec![1, 2]], vec![], applied, "v"),
            node(2, ServerState::Follower, vec![vec![1, 2]], vec![], applied, "v"),
            node(9, ServerState::Leader, vec![vec![1, 2]], vec![], applied, "v"),
        ];
        let converged = check_converged(&snapshots).unwrap();
        assert_eq!(converged.leader, 1);
        assert!(!converged.members.contains(&9));
    }

    /// Put a node's membership at `log_id`: the newest one is the surviving
    /// cluster's.
    fn membership_at(mut node: (NodeId, FullNodeSnapshot), log_id: LogId) -> (NodeId, FullNodeSnapshot) {
        let membership = node.1.raft.membership_config.membership().clone();
        node.1.raft.membership_config = Arc::new(StoredMembership::new(Some(log_id), membership));
        node
    }

    #[test]
    fn removed_leader_with_stale_state_ignored() {
        let applied = Some(log_id(4, 2, 30));
        let stale = Some(log_id(2, 6, 20));
        // Node 6 led term 2 and was removed before it heard of term 4: it
        // still reports itself Leader of {2, 3, 6}, a membership the others
        // replaced.
        let snapshots = vec![
            membership_at(
                node(2, ServerState::Leader, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(4, 2, 25),
            ),
            membership_at(
                node(3, ServerState::Follower, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(4, 2, 25),
            ),
            membership_at(
                node(4, ServerState::Follower, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(4, 2, 25),
            ),
            membership_at(
                node(6, ServerState::Leader, vec![vec![2, 3, 6]], vec![], stale, "old"),
                log_id(2, 6, 10),
            ),
        ];
        let converged = check_converged(&snapshots).unwrap();
        assert_eq!(converged.leader, 2);
        assert_eq!(converged.members, [2, 3, 4].into_iter().collect());
    }

    #[test]
    fn stale_leader_still_a_member_rejected() {
        let applied = Some(log_id(4, 2, 30));
        // Node 3 still claims an old leadership, and the cluster it belongs
        // to has not deposed it yet.
        let snapshots = vec![
            membership_at(
                node(2, ServerState::Leader, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(4, 2, 25),
            ),
            membership_at(
                node(3, ServerState::Leader, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(2, 3, 10),
            ),
            membership_at(
                node(4, ServerState::Follower, vec![vec![2, 3, 4]], vec![], applied, "v"),
                log_id(4, 2, 25),
            ),
        ];
        let err = check_converged(&snapshots).unwrap_err();
        assert!(err.contains("exactly one leader"), "{err}");
    }

    #[test]
    fn learner_leader_accepted() {
        let applied = Some(log_id(2, 1, 10));
        // Node 1 was demoted to learner but keeps leading: openraft only steps
        // down a leader that is removed from the config entirely.
        let snapshots = vec![
            node(1, ServerState::Leader, vec![vec![2, 3]], vec![1], applied, "v"),
            node(2, ServerState::Follower, vec![vec![2, 3]], vec![1], applied, "v"),
            node(3, ServerState::Follower, vec![vec![2, 3]], vec![1], applied, "v"),
        ];
        let converged = check_converged(&snapshots).unwrap();
        assert_eq!(converged.leader, 1);
        assert_eq!(converged.members, [1, 2, 3].into_iter().collect());
    }

    #[test]
    fn joint_membership_rejected() {
        let applied = Some(log_id(2, 1, 10));
        let snapshots = vec![
            node(
                1,
                ServerState::Leader,
                vec![vec![1, 2, 3], vec![1, 2]],
                vec![],
                applied,
                "v",
            ),
            node(
                2,
                ServerState::Follower,
                vec![vec![1, 2, 3], vec![1, 2]],
                vec![],
                applied,
                "v",
            ),
            node(
                3,
                ServerState::Follower,
                vec![vec![1, 2, 3], vec![1, 2]],
                vec![],
                applied,
                "v",
            ),
        ];
        let err = check_converged(&snapshots).unwrap_err();
        assert!(err.contains("joint"), "{err}");
    }

    #[test]
    fn lagging_member_rejected() {
        let applied = Some(log_id(2, 1, 10));
        let behind = Some(log_id(2, 1, 7));
        let snapshots = vec![
            node(1, ServerState::Leader, vec![vec![1, 2]], vec![], applied, "v"),
            node(2, ServerState::Follower, vec![vec![1, 2]], vec![], behind, "v"),
        ];
        let err = check_converged(&snapshots).unwrap_err();
        assert!(err.contains("n2"), "{err}");
    }

    #[test]
    fn missing_member_rejected() {
        let applied = Some(log_id(2, 1, 10));
        let snapshots = vec![node(1, ServerState::Leader, vec![vec![1, 2]], vec![], applied, "v")];
        let err = check_converged(&snapshots).unwrap_err();
        assert!(err.contains("n2"), "{err}");
    }

    #[test]
    fn diverged_state_machine_rejected() {
        let applied = Some(log_id(2, 1, 10));
        let snapshots = vec![
            node(1, ServerState::Leader, vec![vec![1, 2]], vec![], applied, "v1"),
            node(2, ServerState::Follower, vec![vec![1, 2]], vec![], applied, "v2"),
        ];
        let err = check_converged(&snapshots).unwrap_err();
        assert!(err.contains("data differs"), "{err}");
    }
}
