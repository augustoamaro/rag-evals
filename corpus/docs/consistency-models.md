# Consistency Models

A consistency model is a contract between a data store and its clients that defines
which orderings of reads and writes are permitted. Stronger models are easier to
reason about but cost more in latency and availability; weaker models are faster and
more available but push complexity onto the application.

Strong consistency, or linearizability, makes the system behave as if there were a
single copy of the data and every operation took effect instantaneously at some
point between its invocation and its response. Once a write completes, every
subsequent read returns that value.

Eventual consistency only guarantees that, if no new writes are made, all replicas
eventually converge to the same value. In the meantime, different replicas may
return different values for the same key. Eventual consistency is common in
high-availability systems that prioritize uptime over immediate agreement.

Causal consistency sits between the two: operations that are causally related are
seen by all processes in the same order, while concurrent operations may be seen in
different orders. It preserves cause and effect without requiring a global total
order, which makes it cheaper than linearizability while still avoiding the most
confusing anomalies of eventual consistency. Read-your-writes consistency, a useful
session guarantee, ensures a client always sees the effects of its own prior writes.
