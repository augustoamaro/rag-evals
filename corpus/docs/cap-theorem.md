# The CAP Theorem

The CAP theorem states that a distributed data store cannot simultaneously
guarantee all three of consistency, availability, and partition tolerance. Since
network partitions are unavoidable in any real distributed system, the practical
choice during a partition is between consistency and availability: you must pick
one and give up the other.

Consistency here means linearizability — every read observes the most recent
completed write. Availability means every request to a non-failing node receives a
non-error response, even if it might be stale. Partition tolerance means the system
continues to operate despite messages being dropped between nodes.

A CP system favors consistency: during a partition it may reject requests rather
than return possibly stale data. A CP store like a strongly consistent database
sacrifices availability to keep every replica in agreement. An AP system favors
availability: it keeps serving reads and writes on both sides of the partition and
reconciles the divergence afterward, accepting that clients may temporarily read
stale values.

In normal operation, with no partition present, a system can offer both strong
consistency and high availability. CAP only forces the trade-off when a partition
actually occurs, which is why practitioners often describe the real decision as a
spectrum of latency versus consistency rather than a binary choice.
