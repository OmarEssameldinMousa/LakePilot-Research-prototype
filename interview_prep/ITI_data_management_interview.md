# ITI 9-Month Scholarship — Data Management Track
## Technical interview preparation: graduation project drill-down

**How this interviewer works:** he asks one open question, then recursively drills
into every noun you use. Every term you say is a door he can open. So the
strategy is not to say less — it is to **only say things you can defend two
levels deeper.**

Read Part 0 first. Then read Part 13 (the traps), because that is where an
interview is actually won or lost.

---

## Contents

| Part | Topic |
|---|---|
| 0 | The opening answer, and how to control the drill-down |
| 1 | Lakehouse fundamentals |
| 2 | Open table formats and Apache Iceberg |
| 3 | The small-file problem |
| 4 | Compaction |
| 5 | Partitioning, pruning, and partition drift |
| 6 | Parquet and storage internals |
| 7 | Spark and the compute layer |
| 8 | Object storage and catalogs |
| 9 | Your project, specifically |
| 10 | The RL part (keep it short, but be solid) |
| 11 | Classic data-management questions |
| 12 | The tools landscape |
| 13 | Traps and weak points — with honest answers |
| 14 | Questions you ask him |
| 15 | Rapid-fire drill |

---

# Part 0 — The opening answer

## 0.1 The core principle

**Plant doors you want opened.** Your opening answer should contain 4–6 technical
terms that you are *strongest* on. He will pick from what you say. If you say
"we used reinforcement learning to optimise storage," he picks RL. If you say
"we automated compaction and partition evolution on Apache Iceberg," he picks
Iceberg — which is the Data Management answer he wants anyway.

**Do not mention anything you cannot survive two follow-ups on.**

## 0.2 The 30-second version (use this first)

> "My graduation project automates physical maintenance of a data lakehouse.
> When you continuously ingest into an open table format like Apache Iceberg,
> two things degrade over time: you accumulate thousands of small files, and
> your partition layout stops matching how people actually query the table.
> Both are normally fixed by hand or by fixed-threshold scripts. We built a
> system that decides *when* to compact and *when* to change the partition
> specification, based on the live workload, and we measured it against 17
> hand-tuned heuristic policies on a real Iceberg table running on Spark."

That is deliberately dense with data-management hooks: *lakehouse, Iceberg, small
files, partition layout, compaction, partition spec, workload, Spark.* Notice it
does **not** lead with reinforcement learning.

## 0.3 The 90-second version (if he says "go on")

Add, in this order:

1. **The environment.** "We built a simulation harness called LakeGym on top of
   a real stack — Spark 3.5, Iceberg with a REST catalog, MinIO for
   S3-compatible object storage, all in Docker. It's a *real* table: real
   Parquet files, real `rewrite_data_files` compaction, real partition-spec
   evolution, and we *measure* query latency rather than modelling it."
2. **The decision problem.** "At each step we ingest a batch, the policy picks
   one of eight actions — no-op, compact to a 32/64/128 KB target, switch
   partitioning to hour/region/event-type, or drop partitioning — then we run a
   query drawn from the current workload profile and score the result."
3. **The objective.** "The reward combines four things: measured query latency,
   file count, partition pruning ratio, and block utilisation."
4. **The result.** "The best learned policy beat the strongest heuristic by
   about 8% on that objective, and the advantage held on two workloads it had
   never seen. But I'll be honest — it was *not* the cheapest policy; a simple
   threshold heuristic cost less to run."

That last sentence is a power move. It signals you measured honestly, and it
usually earns a "why?" that you can answer well (see 13.4).

## 0.4 Steering technique

When he drills somewhere you're weak, answer the part you know, then **bridge**:

> "…I know it at the spec level rather than the implementation level. What I
> worked with directly was X — [give the strong thing]."

Never bluff. This interviewer's whole method is depth-testing; a bluff collapses
on the second follow-up and costs you more than "I don't know that part."

**The honest-boundary sentence:** *"I haven't worked with that directly — my
understanding is [one sentence]. Is that right?"* This converts a gap into a
conversation and often impresses more than a shaky answer.

---

# Part 1 — Lakehouse fundamentals

### Q: What is a data lakehouse?

An architecture that puts **warehouse-style guarantees on lake-style storage**.
You keep data as open-format files (Parquet) on cheap object storage, and add a
**metadata layer** (Iceberg / Delta / Hudi) that provides ACID transactions,
schema evolution, time travel, and partition management. You get the warehouse's
correctness and the lake's cost, openness, and scale — and BI and ML can read the
same single copy.

### Q: How is it different from a data warehouse?

| | Warehouse | Lakehouse |
|---|---|---|
| Storage | Proprietary internal format, coupled to the engine | Open format (Parquet) on object storage |
| Compute | Coupled to storage | Decoupled — scale independently |
| Schema | Schema-on-write | Schema-on-write, but evolvable cheaply |
| Workloads | BI / SQL | BI, SQL, ML, streaming on one copy |
| Data types | Structured | Structured + semi/unstructured |
| Cost | High ($/TB, licensed) | Object storage cost + compute |
| Lock-in | High | Low — any engine reading the spec |

**The one-liner:** *"A warehouse owns its storage. A lakehouse doesn't — the
table format does, and it's an open spec."*

### Q: How is it different from a data lake?

A data lake is just files in object storage. No transactions, no schema
enforcement, no consistent view during concurrent writes, no way to safely
update or delete a row. That's how lakes become **data swamps**. The lakehouse
adds the metadata/transaction layer on top of exactly the same files.

Concretely, on a plain lake you cannot: run a `DELETE` safely, get a consistent
read while a writer is committing, roll back a bad load, or evolve a schema
without breaking readers.

### Q: Why did the lakehouse appear? What problem drove it?

The **two-tier architecture** problem. Companies ran a lake for raw/ML data *and*
a warehouse for BI, and copied data between them via ETL. That means: data
duplicated (2× storage cost), staleness between the two, two governance models,
two security models, and constant reconciliation bugs ("why doesn't the dashboard
match the notebook?"). The lakehouse collapses both tiers onto one copy.

### Q: What are the downsides of a lakehouse?

Be ready — this shows maturity:
- **Maintenance is now your job.** Compaction, snapshot expiry, orphan-file
  cleanup — a warehouse does this invisibly. *(This is literally my project.)*
- Latency for small point-lookups is worse than a tuned warehouse or an OLTP DB.
- Higher operational complexity: catalog, engine, format versions must be
  compatible.
- Less mature governance/fine-grained security than established warehouses,
  though catalogs are closing the gap.
- Concurrency: optimistic concurrency means write-heavy multi-writer workloads
  can hit commit retries.

### Q: What is a medallion architecture?

Bronze → Silver → Gold layering. **Bronze** = raw ingested data, append-only,
immutable, schema-on-read-ish. **Silver** = cleaned, deduplicated, conformed,
joined. **Gold** = business-level aggregates / marts serving BI. It is a
convention for organising lakehouse tables, not a technology.

---

# Part 2 — Open table formats and Apache Iceberg

### Q: What is an open table format?

A **specification** for how to represent a table as a set of files plus metadata
files on object storage — such that any engine implementing the spec can read
and write it correctly and transactionally. It defines what a table *is*
(schema, partitioning, snapshots, file lists with statistics) independently of
any engine.

The key insight: before this, "table" was a concept owned by an engine. Now it's
a concept owned by an open file layout.

### Q: What are the main open table formats and how do they compare?

| | **Iceberg** | **Delta Lake** | **Hudi** |
|---|---|---|---|
| Origin | Netflix → Apache | Databricks → Linux Foundation | Uber → Apache |
| Metadata model | Snapshot tree: metadata file → manifest list → manifests | Ordered transaction log (JSON) + checkpoints (Parquet) | Timeline + file groups/slices |
| Partitioning | **Hidden partitioning + partition evolution** | Explicit columns; liquid clustering (newer) | Explicit |
| Update strategy | CoW and MoR | CoW and MoR (deletion vectors) | **Strong upsert focus**, CoW & MoR |
| Best at | Huge tables, engine neutrality, schema/partition evolution | Databricks ecosystem, simplicity, streaming | Record-level upserts, CDC, incremental pulls |
| Engine support | Broadest (Spark, Flink, Trino, Dremio, Snowflake, BigQuery, DuckDB) | Broad, Databricks-centric | Narrower |

**If asked "which is best?"** — There's no absolute answer; it depends on
workload. Iceberg for engine-neutrality and evolution-heavy analytics, Hudi for
heavy record-level upserts/CDC, Delta if you're in Databricks. Note the industry
is converging (Delta UniForm, Iceberg REST catalog as a standard).

### Q: Explain Iceberg's architecture. ← **learn this cold, it's the most likely deep-dive**

Four layers, top to bottom:

```
Catalog              → pointer to the CURRENT metadata file  (the only mutable thing)
  ↓
Metadata file (.json) → schema(s), partition spec(s), sort orders,
                        snapshot list, current snapshot id, table properties
  ↓
Manifest list (avro)  → ONE per snapshot. Lists the manifest files in that
                        snapshot, each with partition value ranges + row counts
  ↓
Manifest file (avro)  → lists DATA FILES, each with: path, partition values,
                        record count, file size, and per-column stats
                        (min, max, null count, value count)
  ↓
Data files            → Parquet (or ORC/Avro)
```

**Why this shape matters:** query planning is a **metadata tree walk**, not a
directory listing. To plan `WHERE ts BETWEEN a AND b`, the engine prunes at the
manifest-list level (partition ranges), then at the manifest level (per-file
min/max), and only then opens data files. On a million-file table you might open
a handful of metadata files instead of doing a `LIST` over a million objects.

### Q: How does Iceberg give you ACID on object storage?

Object stores have **no atomic rename** and no multi-object transaction. Iceberg
sidesteps this: a commit is an **atomic compare-and-swap of a single pointer** in
the catalog.

The write path:
1. Write new data files (nobody sees them — they're not referenced yet).
2. Write new manifest(s) and a new manifest list.
3. Write a new metadata file describing the new snapshot.
4. **Atomically swap** the catalog's current-metadata pointer from `v_n` to
   `v_n+1`, conditional on it still being `v_n`.

If another writer committed in between, the CAS fails, and you **retry** —
re-reading the new base state and re-applying. That's **optimistic concurrency
control**. Readers always resolve the pointer once and see one immutable
snapshot → **snapshot isolation**, and no reader ever sees a half-written commit.

### Q: What does "engine-agnostic" mean, and why does it matter?

The table's definition lives in **open-spec files in your own object storage**,
not inside a vendor's engine. Any engine that implements the Iceberg spec —
Spark, Flink, Trino, Dremio, Snowflake, BigQuery, DuckDB, Athena — can read and
write the *same* table, transactionally, with no export/import and no copy.

Why it matters commercially: **no vendor lock-in**, and you can use the right
engine per job (Flink to stream-ingest, Spark to batch-transform, Trino for
interactive BI) against one physical copy. Contrast with a warehouse, where
leaving means physically exporting everything.

### Q: What is a snapshot? What is time travel?

A snapshot is a **complete, immutable list of all data files** constituting the
table at one commit. Every commit creates a new snapshot; old ones remain valid
until expired. So you can query the table *as of* a timestamp or snapshot ID:

```sql
SELECT * FROM db.tbl VERSION AS OF 3821550127947089009;
SELECT * FROM db.tbl TIMESTAMP AS OF '2026-08-01 10:00:00';
```

**Uses:** reproduce a report, debug "what changed," audit, and — the one people
actually rely on — **roll back a bad write** instantly instead of restoring from
backup. I used snapshot rollback directly in my project to implement an oracle
that trials an action and undoes it.

### Q: What's the cost of keeping snapshots? How do you manage it?

Old snapshots pin old data files, so storage grows and metadata bloats. You run
maintenance:
- `expire_snapshots` — drop snapshots older than a retention window and delete
  files no longer referenced by *any* live snapshot.
- `remove_orphan_files` — delete files in the table location that no metadata
  references at all (failed writes).
- `rewrite_manifests` — re-cluster manifest entries so planning stays fast.

**Trap:** expiring snapshots destroys your ability to time-travel to them. Set
retention from your actual audit/rollback requirement.

### Q: What is schema evolution in Iceberg, and why is it safe?

You can add, drop, rename, reorder, and widen columns without rewriting data.
It's safe because Iceberg tracks columns by **unique field ID**, not by name or
position. A rename changes the name attached to an ID; a drop stops projecting
that ID. Old Parquet files are still read correctly because the mapping is by ID.

This is exactly what breaks in naive Hive-style tables, where a rename or a
reorder silently misaligns columns against old files.

---

# Part 3 — The small-file problem

### Q: What is the small-file problem, and why does it happen?

**Why it happens** — several causes compound:
1. **Streaming / micro-batch ingestion.** Each commit must write at least one
   file per partition per writer task. Commit every 30 s and you generate files
   at that rate, forever.
2. **Parallelism × partition cardinality.** With 200 Spark tasks writing into
   50 partitions, one commit can create up to 10,000 files.
3. **Latency pressure.** Wanting fresh data means committing often, which means
   smaller files. Freshness and file size are in direct tension.
4. **Upserts/deletes in MoR mode** add delete files alongside data files.
5. **Over-partitioning.** Partitioning by an over-granular key (hour, or worse,
   user ID) guarantees tiny files per partition.

### Q: Why is it actually bad? What breaks?

Give the layered answer — this is where you show depth:

- **Metadata / planning.** Every file is a manifest entry with stats. Millions
  of files → huge manifests → query *planning* becomes slow before a single byte
  of data is read.
- **Object-store request cost and latency.** Reading 10,000 small files means
  ~10,000 `GET`s. On S3 each has fixed latency (tens of ms) and a per-request
  price. Throughput collapses to request-overhead-bound.
- **Compression and encoding efficiency.** Parquet's dictionary/RLE encodings
  work per row group. If a file holds 50 rows, the dictionary overhead exceeds
  the savings — you lose the columnar advantage and files are larger *per row*.
- **Task scheduling overhead.** Spark schedules roughly one task per file split.
  With 10,000 tiny files you pay scheduling and JVM overhead per file, and the
  actual work per task is trivial. The cluster spends its time on bookkeeping.
- **Footer overhead.** Every Parquet file has a footer that must be read to plan
  — a fixed cost amortised over the file's rows. Tiny files never amortise it.

**Memorable version:** *"You lose the two things that make columnar analytics
fast — sequential large reads and effective encoding — and you replace them with
per-file fixed costs."*

### Q: What's a good target file size?

Typically **128 MB – 1 GB**, most commonly ~512 MB, aligned with the Parquet row
group size and the engine's split size. It's a trade-off: bigger files mean
better scan throughput and cheaper metadata, but coarser pruning granularity and
more write amplification when you update.

*(In my project the tables were deliberately tiny — 4–5 MB, targets of
32/64/128 KB — because I needed hundreds of full episodes to run in hours, not
weeks. I'll flag that as a limitation; see 13.1.)*

---

# Part 4 — Compaction

### Q: What is compaction?

Rewriting many small data files into fewer, larger ones, **without changing the
logical content of the table**. In Iceberg it's `rewrite_data_files`. It commits
as a normal snapshot: new big files added, old small files removed from the new
snapshot (but still present on disk until snapshots expire).

### Q: What kinds of compaction are there?

1. **Bin-pack** — just pack small files up to a target size. No reordering.
   Cheapest. Fixes the small-file problem only.
2. **Sort / order** — rewrite sorted by one or more columns. This *tightens the
   min/max statistics per file*, which directly improves file-level pruning for
   range predicates on the sort column.
3. **Z-order (multi-dimensional clustering)** — interleaves the bits of several
   columns so that files are clustered in more than one dimension at once. Use
   when queries filter on several columns unpredictably. Weaker per-column
   pruning than a straight sort, but much better *average* across a mixed
   workload.

### Q: What does compaction cost?

- **Read + write amplification** — you read all the small files and write the
  data again. At production scale a compaction is a distributed job occupying
  many executors for minutes.
- **Storage** — old files stay until snapshots expire, so you temporarily hold
  both copies.
- **Commit conflict risk** — a long compaction can conflict with concurrent
  writers and need retry.

### Q: When should you compact? What's the trade-off?

The real question of my project. Too rarely: queries degrade, metadata bloats.
Too often: you burn compute and I/O rewriting data that was about to be rewritten
anyway, and each rewrite is pure cost with no logical change.

Standard practice is a **threshold heuristic** — compact when file count in a
partition exceeds N, or when the fraction of files below the target size exceeds
some ratio. My project's premise was that a fixed threshold can't be right across
a workload whose ingestion rate and query mix both change over time.

### Q: What other maintenance operations does a lakehouse need?

`expire_snapshots`, `remove_orphan_files`, `rewrite_manifests`, and for MoR
tables, **compacting delete files** into the base data. Also refreshing table
statistics. A production Iceberg deployment needs all of these scheduled — this
is the operational burden that a warehouse hides from you.

### Q: Copy-on-write vs merge-on-read?

- **CoW:** an update rewrites the entire data file(s) containing the affected
  rows. Reads are fast (no merge). Writes are expensive. Good for read-heavy,
  infrequent-update tables.
- **MoR:** an update writes a small **delete file** (position deletes = "row 47
  of file X is dead", or equality deletes = "all rows where id=5 are dead") and
  optionally new data. Writes are fast. Reads must merge base + deletes, so they
  degrade until you compact. Good for streaming/CDC.

Note that MoR converts the small-file problem into a **small-delete-file
problem** — you still need compaction, just of a different thing.

---

# Part 5 — Partitioning, pruning, and partition drift

### Q: What is partitioning and why do it?

Physically grouping rows by the value of some column (or a transform of it) so
that a query filtering on that column can **skip** entire groups without reading
them. It is the single biggest lever on scan volume in analytics.

### Q: What is partition pruning?

At planning time the engine compares the query predicate against the partition
values recorded in the metadata, and eliminates files that cannot match — before
reading any data. `WHERE region = 'EU'` on a region-partitioned table reads only
EU files.

Iceberg prunes at multiple levels: manifest list (partition ranges per manifest)
→ manifest (per-file partition values and column min/max) → within the Parquet
file (row-group stats, page index). **This is why Iceberg tables plan fast at
scale.**

### Q: What is hidden partitioning? ← **strong Iceberg-specific answer**

In Hive, if you partitioned by day you had to create a physical `dt` column and
users had to write `WHERE dt = '2026-08-09'`. If they wrote `WHERE ts >
'2026-08-09'` instead, they got a **full table scan** — correct results, terrible
performance, and no warning.

Iceberg stores a **partition transform** (`days(ts)`, `bucket(16, id)`,
`truncate(10, name)`) in the partition spec. Users query the *natural* column
`ts`, and Iceberg derives the partition predicate itself. Consequences:
- No redundant column to maintain and keep consistent.
- Users cannot accidentally miss the partition filter.
- **The physical layout is decoupled from the query text** — which is precisely
  what makes partition evolution possible.

### Q: What is partition evolution?

Changing the partition spec of an existing table **without rewriting historical
data**. Old files keep their old spec; new writes use the new spec. Each manifest
records which spec-ID its files were written under, and the planner handles both
simultaneously.

Only possible *because* of hidden partitioning: since queries reference logical
columns rather than physical partition columns, changing the physical layout
doesn't break any query.

*(This is one of the two action families in my project: the agent can switch the
table from hour-partitioning to region-partitioning at runtime.)*

### Q: What is partition drift? Why does it occur?

The layout stops matching how the table is actually queried. Causes:
- **Query workload shifts.** You partitioned by `region` because dashboards
  filtered by region; six months later the main consumer is a time-series job
  filtering by timestamp. Same table, now badly laid out.
- **Data distribution shifts** → **partition skew**: one region becomes 80% of
  volume, so its partitions are huge and the parallelism is unbalanced, while
  other partitions are tiny (small-file problem again).
- **Cardinality growth**: a key that had 100 distinct values now has 100,000, so
  partitioning by it produces tiny partitions.

**The trap of fixing it:** changing the spec only affects *new* data. Old data is
still laid out the old way — you only get the benefit once the data is
**rewritten** into the new layout, i.e. by compaction. This coupling was my
project's clearest measured result (see 9.6).

### Q: How do you choose a partition key?

- Match the **dominant filter predicate** of the workload.
- Target partition sizes in the **hundreds of MB to a few GB** — enough that
  files are large, few enough that metadata stays small.
- **Low-to-moderate cardinality**, or use `bucket(N, col)` to bound it.
- Avoid partitioning on a high-cardinality key directly (user_id) — that's the
  classic over-partitioning mistake.
- Watch for **skew** in the key distribution.

### Q: Partitioning vs bucketing vs clustering vs indexing?

- **Partitioning** — physical separation by value; enables skipping whole
  directories/file groups. Coarse.
- **Bucketing** — hash into a fixed number of buckets; bounds cardinality and
  enables join co-location (shuffle-free joins on the bucket key).
- **Clustering / sorting / Z-order** — ordering *within* files so min/max stats
  are tight; enables fine-grained file skipping without a directory structure.
- **Indexing** — an auxiliary structure (B-tree, bloom filter, bitmap). In
  lakehouses this is mostly min/max stats + bloom filters in the Parquet footer,
  not a classical index.

---

# Part 6 — Parquet and storage internals

### Q: What is Parquet? Why columnar?

A binary **columnar** file format. Values of the same column are stored
contiguously.

Why that's fast for analytics:
1. **Projection pushdown** — reading 3 columns of 200 reads ~1.5% of the bytes.
2. **Compression ratio** — adjacent values in a column are similar and often
   repeat, so dictionary/RLE/delta encoding is dramatically more effective than
   on a row of mixed types.
3. **Vectorised execution** — a column chunk is a typed array, so the engine
   processes batches in tight loops (and SIMD) rather than row-by-row.
4. **Predicate pushdown** — footer statistics let you skip row groups/pages
   entirely.

Row formats (Avro, CSV, JSON) win when you need whole records, write-heavy
append, or streaming — hence **Avro for Iceberg's manifests** (small, whole-record
reads, schema evolution) and **Parquet for the data**.

### Q: Describe Parquet's internal layout.

```
File
 ├── Row Group 1  (a horizontal slice, typically ~128 MB)
 │    ├── Column Chunk A → Pages (each with its own stats + encoding)
 │    ├── Column Chunk B → Pages
 │    └── ...
 ├── Row Group 2
 └── Footer  ← schema, row-group metadata, per-column min/max/null counts,
                offsets, and the magic bytes
```

The footer is at the **end**, so a reader does: read last 8 bytes → get footer
length → read footer → plan → issue targeted range `GET`s for exactly the pages
it needs. That's why a small file is inefficient: you pay footer read + planning
for almost no data.

### Q: How does predicate pushdown work in Parquet?

The footer holds min/max per column per row group. For `WHERE x > 100`, the
reader compares against each row group's max for `x`; if max ≤ 100 the entire row
group is skipped without decompression. The **page index** (newer) extends this to
page granularity. Bloom filters can be written for equality predicates on
high-cardinality columns.

**Key insight to state:** this only works if the data is **sorted or clustered**
on the predicate column. On randomly ordered data, every row group's min/max
spans nearly the full range, so nothing is skipped — the stats exist but are
useless. That's precisely why *sort* compaction and Z-ordering matter, not just
bin-pack.

### Q: Parquet vs ORC vs Avro?

- **Parquet** — columnar, the de-facto analytics standard, broadest support.
- **ORC** — columnar, Hive-native heritage, strong lightweight indexes and
  built-in ACID support in Hive; comparable performance.
- **Avro** — row-based, excellent schema evolution, compact binary; used for
  streaming (Kafka) and for Iceberg's own metadata.

### Q: Which compression codec?

- **Snappy** — fast compress/decompress, moderate ratio. Historic default.
- **Zstd** — better ratio at comparable speed; increasingly the recommended
  default.
- **Gzip** — high ratio, slow. Archival.
- **LZ4** — fastest, lowest ratio.

Trade-off is CPU vs I/O vs storage. On object storage where network is the
bottleneck, a better ratio (zstd) usually wins.

---

# Part 7 — Spark and the compute layer

### Q: Explain Spark's architecture.

**Driver** (runs your code, builds the logical plan, schedules) → **Cluster
manager** (YARN / Kubernetes / standalone) → **Executors** (JVM processes that
run tasks and hold cached data).

Execution hierarchy: **Job** (per action) → **Stages** (split at shuffle
boundaries) → **Tasks** (one per partition, the unit of parallelism).

### Q: Narrow vs wide transformations?

- **Narrow** — each output partition depends on one input partition (`map`,
  `filter`, `union`). No data movement; pipelined within a stage.
- **Wide** — output partitions depend on many input partitions
  (`groupBy`, `join`, `distinct`, `repartition`). Requires a **shuffle**: write
  to disk, transfer over network, read. Shuffles define stage boundaries and are
  the main cost in most Spark jobs.

### Q: What is a shuffle and why is it expensive?

Data is repartitioned across executors by key. It involves serialisation, disk
spill on the map side, network transfer, and deserialisation + merge on the
reduce side. It's expensive because it's the only part of Spark that is
simultaneously disk-bound, network-bound, and a synchronisation barrier.

### Q: What is data skew and how do you fix it?

One key has far more rows than others, so one task runs for hours while the rest
finish in seconds. Fixes:
- **Salting** — append a random suffix to the hot key, aggregate in two passes.
- **Broadcast join** — if one side is small, broadcast it and avoid the shuffle
  entirely.
- **AQE skew join optimisation** — Spark 3 detects skewed partitions at runtime
  and splits them automatically.
- Filter/handle the hot key separately.

### Q: What is Catalyst? What is AQE?

**Catalyst** is Spark's query optimiser: parse → analyse (resolve against
catalog) → logical optimisation (predicate pushdown, projection pruning, constant
folding) → physical planning (choose join strategies) → code generation
(Tungsten produces JVM bytecode for tight loops).

**AQE (Adaptive Query Execution)** re-optimises *at runtime* using actual
statistics observed mid-query: coalescing shuffle partitions, converting
sort-merge joins to broadcast joins when a side turns out small, and splitting
skewed partitions. It exists because static cost estimates on a lake are usually
wrong.

### Q: `repartition` vs `coalesce`?

`repartition(n)` does a **full shuffle** and can increase or decrease partition
count, producing even sizes. `coalesce(n)` only **decreases** and avoids a full
shuffle by merging adjacent partitions — cheap, but can leave uneven partitions.
Use `coalesce` to reduce output file count cheaply; use `repartition` when you
need balance or a specific distribution.

### Q: Where does Iceberg fit with Spark?

Iceberg provides a Spark **catalog implementation** and a DataSource V2
connector. Spark asks Iceberg to plan the scan (Iceberg does the metadata tree
walk and returns exactly the file splits needed), then Spark executes. Iceberg
also exposes maintenance as **stored procedures**:
`CALL catalog.system.rewrite_data_files(...)`, `expire_snapshots`, etc.

In my stack the Spark configuration registers
`org.apache.iceberg.spark.SparkCatalog` with `type=rest`, `uri=http://rest:8181`,
`io-impl=S3FileIO`, and `warehouse=s3://warehouse/`.

---

# Part 8 — Object storage and catalogs

### Q: Object storage vs a filesystem — why does it matter here?

Object storage (S3, GCS, Azure Blob, MinIO) is a **flat key–value store**, not a
hierarchy. Consequences that shape lakehouse design:
- **No directories.** "Folders" are just key prefixes; `LIST` on a prefix is an
  O(n) paginated API call, not a cheap directory read.
- **No atomic rename.** A rename is copy + delete — non-atomic and O(size). This
  is why Hive's rename-based commit protocol was unsafe, and why Iceberg commits
  by swapping a pointer instead.
- **Immutable objects.** You replace, never edit in place. Which is exactly why
  updates require either rewriting files (CoW) or delete files (MoR).
- **Per-request latency and cost.** Favours few large reads. Reinforces
  everything in Part 3.

### Q: What is MinIO?

An S3-API-compatible object store you can run yourself. In my project it stands
in for S3 inside Docker so the whole stack runs locally with the real S3
semantics and API — same `S3FileIO` code path as production.

### Q: What is a catalog, and what are the options?

The catalog maps a table name to the **location of its current metadata file**,
and provides the **atomic swap** primitive that makes commits transactional. It
is the only mutable pointer in the system.

Options: **Hive Metastore** (legacy, ubiquitous), **AWS Glue**, **JDBC**,
**Nessie** (git-like branching/tagging of data), **REST catalog** (an open API
spec — this is what I used), and vendor implementations like **Polaris** and
**Unity Catalog**.

### Q: Why is the REST catalog significant?

It defines catalog operations as an **open HTTP API** rather than requiring every
engine to embed a specific client library. So the catalog becomes a swappable
service, engines only need an HTTP client, and the server can centrally enforce
authorisation and commit logic. It's the piece that makes multi-engine access
practical rather than theoretical.

---

# Part 9 — Your project, specifically

## 9.1 The stack (be exact — he may ask versions)

Spark 3.5.7 + PySpark, Apache Iceberg via **REST catalog**, **MinIO** for
S3-compatible storage, all orchestrated in Docker Compose. Python 3.10.
TensorFlow 2.20 for the models. The table: 8 columns (`device_id`, `timestamp`,
`temperature`, `humidity`, `region`, `event_type`, `sensor_value`, `payload`),
~200 bytes/row — an IoT-style sensor telemetry table.

## 9.2 What is "LakeGym"?

A simulation environment wrapping the real stack in a step-based loop. Each step:
ingest a batch → policy picks an action → execute the action against the real
Iceberg table → run a query drawn from the current workload profile → **measure**
its latency → compute reward → build the next observation.

**Be precise about what is real vs simulated** (he will respect this):
- **Real:** the Iceberg table, Parquet files, `rewrite_data_files` compaction,
  partition-spec evolution, and the query latency, which is *measured* by
  actually executing the query on Spark.
- **Simulated:** the data (synthetic), the query set (5 templates), the scale
  (4–5 MB tables), and the pruning benefit (applied from an assumed lookup table
  rather than measured from Spark scan metrics).

## 9.3 The action space (8 actions)

`NOOP`; `COMPACT_32KB` / `COMPACT_64KB` / `COMPACT_128KB`;
`PARTITION_HOUR` / `PARTITION_REGION` / `PARTITION_EVENT_TYPE`;
`REMOVE_PARTITION`.

## 9.4 The workload

Five query templates: `TIME_RANGE`, `REGION_FILTER`, `SENSOR_LOOKUP`,
`TYPE_FILTER`, `FULL_SCAN`. Five workload **profiles** that each weight those
templates differently (e.g. `time_heavy` = 60% TIME_RANGE), shifting through the
episode — that's the non-stationarity the system has to react to. Ingestion rate
also varies in six phases, from 35 to 225 rows/step, including a burst.

Deliberately, the ingestion phases and the query-profile phases have **different
boundaries**, so storage pressure and query-mix shift aren't confounded.

## 9.5 The reward

Four components: measured query latency (negative), file count (negative),
partition pruning ratio (positive), block utilisation (positive), weighted
0.30 / 0.20 / 0.30 / 0.20.

**If he asks "why those weights?"** — Be honest: they were chosen to put the four
signals on a comparable scale, not derived from a cost measurement. That's why we
ran a sensitivity analysis perturbing each by ±25% — the policy ranking was
stable (Spearman ρ ≥ 0.95).

## 9.6 The results worth quoting

- Best learned policy beat the best of 17 heuristics by **+8.3%**
  (0.2105 vs 0.1943), paired permutation **p = 0.001**, large effect size, and
  **all 5 independent training runs beat it**.
- Advantage held on **two held-out workloads** (gradual drift; concurrent mixed
  profiles).
- **The ablation is the best data-management finding.** Partition-only (no
  compaction) achieved pruning of just **0.080**, versus **0.225** for the same
  partition logic *with* compaction. Why: declaring a new partition spec doesn't
  move existing data. You only realise the pruning benefit once compaction
  **rewrites** data into the new layout. So compaction and partitioning are not
  independent knobs — and any system that triggers them on separate thresholds
  leaves that interaction on the table.
- Honest: RL was **not** the cheapest. A `Threshold10_C128` heuristic cost $8.37
  vs $8.79 per episode, because it compacted less often.

**If you remember one result, remember the ablation.** It's a pure
data-management insight, it's counter-intuitive, and it's yours.

## 9.7 "Why automate lakehouse maintenance?" ← almost certain question

Layer it:

1. **It's unavoidable work.** A warehouse hides maintenance; a lakehouse hands it
   to you. Compaction, snapshot expiry, orphan cleanup, layout choice — someone
   must own it.
2. **Current practice is fixed thresholds**, which encode an assumption that the
   workload is stationary. Real workloads aren't: ingestion rate changes, query
   mix changes, cardinality grows.
3. **The two decisions interact** (see 9.6), so tuning them independently is
   leaving value behind.
4. **It's genuinely a sequential decision problem**: maintenance is an
   *investment* — you pay compute now for query benefit later, and today's action
   changes tomorrow's state. That's exactly the shape RL addresses, and it's
   what a threshold rule structurally cannot represent.
5. **Cost.** At scale, compaction is a distributed rewrite job. Doing it too
   often is expensive; too rarely degrades every query. The optimum moves.

## 9.8 "What would you do differently?"

Strong, honest answer:
- **Measure pruning empirically** by instrumenting Spark's scan metrics
  (files-skipped per query) instead of using an assumed lookup table.
- **Run at realistic scale** — GB-to-TB tables, so the latency/file-count
  relationship reflects data volume rather than per-file overhead.
- **Test more schemas** — my partition findings depend on the cardinality of the
  partitioning column.
- Compare against **production auto-optimisers** (Databricks OPTIMIZE, Snowflake
  auto-clustering) on an identical workload.

---

# Part 10 — The RL part (keep it brief, but be solid)

He's a Data Management interviewer. Don't volunteer depth here — but don't be
caught empty either.

### Q: Why reinforcement learning and not just supervised learning / a rule?

Because there's **no labelled ground truth** for "the right maintenance action at
this moment," and because the payoff is **delayed and state-changing**: compacting
now costs compute now and pays off across many future queries, and it changes the
state the next decision is made from. Supervised learning needs labels;
optimisation needs a static objective. Sequential decision-making under delayed
reward is the definition of RL.

### Q: What is the architecture?

Hierarchical: a **rule-based meta-controller** decides *whether* to act and which
specialist to delegate to (based on urgency scores from file count and pruning
ratio), and two **learned specialists** decide *what* to do — one owns compaction
actions, one owns partition actions.

Say plainly: **the arbitration tier is not learned; the specialists are.** Being
precise about that is better than overclaiming "end-to-end deep RL."

### Q: Why hierarchical rather than one agent?

Three reasons: the specialists get **focused reward signals** (compaction reward
vs partition reward) instead of one diluted signal; the action space **factorises**
(adding a new maintenance type adds a specialist, not a Cartesian product); and
the urgency logic is **auditable** — you can log why the system chose to act,
which matters operationally.

### Q: Which algorithms?

Three architectures compared under matched configuration: an attention-based
actor-critic, a feedforward actor-critic (both trained with offline PPO-clip),
and a **dueling Double DQN**. DDQN performed best. Training was **offline**, from
logged trajectories of heuristic policies — I never trained online against the
live table, because each episode takes ~25–35 minutes of real Spark execution.

### Q: One-line definitions if pressed

- **Q-value** — expected cumulative future reward from taking action *a* in state
  *s* and behaving optimally after.
- **Double DQN** — uses the online network to *select* the next action and the
  target network to *evaluate* it, which reduces the overestimation bias of
  vanilla DQN's single `max`.
- **Dueling** — splits the network into a state-value stream V(s) and an
  advantage stream A(s,a), which helps when many actions have similar value (here:
  most of the time doing nothing is fine).
- **Offline RL** — learning from a fixed dataset of logged trajectories without
  interacting with the environment during training.

---

# Part 11 — Classic data-management questions

He *will* ask general track questions. Have these tight.

### OLTP vs OLAP

| | OLTP | OLAP |
|---|---|---|
| Purpose | Run the business | Analyse the business |
| Ops | Many small reads/writes | Few large scans/aggregations |
| Schema | Highly normalised (3NF) | Denormalised (star schema) |
| Storage | Row-oriented | Column-oriented |
| Metric | Transactions/sec, latency | Throughput, scan volume |

### ACID

- **Atomicity** — all or nothing.
- **Consistency** — constraints hold before and after.
- **Isolation** — concurrent transactions don't see each other's intermediate
  state.
- **Durability** — committed data survives failure.

*In Iceberg:* atomicity and isolation come from the atomic pointer swap and
immutable snapshots; durability from object storage; consistency from schema and
the commit protocol.

### Normalisation

- **1NF** — atomic values, no repeating groups.
- **2NF** — 1NF + no partial dependency on part of a composite key.
- **3NF** — 2NF + no transitive dependency (non-key → non-key).
- **BCNF** — every determinant is a candidate key.

**Why normalise?** Eliminate update/insert/delete anomalies and redundancy.
**Why denormalise for analytics?** Joins are expensive at scan scale; analytical
data is append-mostly so update anomalies barely matter. You trade write-side
integrity for read-side speed — deliberately.

### Star vs snowflake schema

- **Star** — one central **fact** table (measures + foreign keys) surrounded by
  denormalised **dimension** tables. Fewer joins, faster, simpler for BI tools.
- **Snowflake** — dimensions normalised into sub-dimensions. Less redundancy,
  more joins, more complex.

Star usually wins in practice. **Grain** = what one fact row represents — define
it first, always.

### Fact table types

**Transaction** (one row per event), **periodic snapshot** (state at regular
intervals), **accumulating snapshot** (one row per process instance, updated as
it progresses). Measures are **additive** (sum over all dimensions),
**semi-additive** (sum over some — e.g. account balance, not over time), or
**non-additive** (ratios, percentages).

### Slowly Changing Dimensions

- **Type 0** — never change.
- **Type 1** — overwrite; no history.
- **Type 2** — new row per change with validity dates / current flag; **full
  history**. Most common.
- **Type 3** — add a "previous value" column; limited history.

*Lakehouse angle:* Type 2 is a **MERGE** workload — heavy upserts, which is
exactly the CoW/MoR trade-off and the reason Hudi exists.

### ETL vs ELT

**ETL** transforms before loading (transform on a separate engine) — historically
required because warehouse storage/compute were expensive. **ELT** loads raw then
transforms in-place using the warehouse/lakehouse's own compute — enabled by cheap
object storage and elastic compute, and preferred now because you keep the raw
data and can re-derive transformations (which is exactly the Bronze layer idea).

### CDC (Change Data Capture)

Capturing row-level changes from a source database — typically by reading the
transaction log (e.g. Debezium reading MySQL binlog / Postgres WAL) — and
streaming them downstream. Log-based is preferred over query-based (polling
`updated_at`) because it catches deletes, doesn't miss intra-poll changes, and
doesn't load the source. Downstream this becomes a MERGE/upsert workload.

### CAP theorem

Under a network **partition**, you must choose **consistency** or
**availability**. Not "pick 2 of 3" — partitions aren't optional. Related:
**BASE** (Basically Available, Soft state, Eventually consistent) as the
AP-leaning alternative to ACID.

### Indexing

**B-tree** (ordered, ranges and equality, the default), **Hash** (equality only),
**Bitmap** (low-cardinality, analytical), **Clustered** (defines physical row
order — one per table) vs **Non-clustered** (separate structure with pointers).

*Lakehouse angle:* there are no classical indexes — the equivalent is
**min/max statistics + bloom filters** in Parquet footers and Iceberg manifests,
which only work if data is clustered on the predicate column.

### Data quality dimensions

Accuracy, completeness, consistency, timeliness, validity, uniqueness. Enforced
via constraints, expectations frameworks (Great Expectations, dbt tests), and
reconciliation checks.

### Data governance terms

**Data lineage** (where data came from and what transformed it), **data
catalog** (searchable metadata inventory), **data steward** (accountable owner),
**master data management**, **PII / GDPR right-to-erasure** — note the last one is
genuinely hard on immutable files, and is a real driver for MoR delete files and
for snapshot expiry policies.

---

# Part 12 — The tools landscape

Know what each is *for*, one line each. He may ask "have you used X?" — answer
honestly, but show you know where it sits.

| Category | Tools | One-line purpose |
|---|---|---|
| Table formats | Iceberg, Delta, Hudi | ACID + metadata over files |
| File formats | Parquet, ORC, Avro | How bytes are laid out |
| Processing | Spark, Flink, Trino, Presto, Dremio | Distributed compute/query |
| Streaming | Kafka, Kinesis, Pulsar | Durable event log |
| Orchestration | Airflow, Dagster, Prefect | Schedule + dependency DAGs |
| Transformation | dbt | SQL-based ELT modelling + tests + lineage |
| Ingestion / CDC | Debezium, Airbyte, Fivetran, NiFi | Source → lake |
| Warehouses | Snowflake, BigQuery, Redshift | Managed OLAP |
| Catalogs | Hive Metastore, Glue, Nessie, Polaris, Unity | Table registry + commit |
| Storage | S3, GCS, ADLS, MinIO, HDFS | Where files live |
| Quality | Great Expectations, Soda, dbt tests | Assertions on data |

### Q: Batch vs streaming?

Batch = bounded data, high throughput, simpler, higher latency. Streaming =
unbounded, low latency, must handle **late/out-of-order events** (watermarks,
windowing) and **exactly-once** semantics. **Lambda architecture** runs both and
merges; **Kappa** runs streaming only and replays the log for reprocessing. The
lakehouse blurs this — you can stream-ingest into an Iceberg table and batch-read
it.

### Q: What is idempotency and why does it matter in pipelines?

Re-running the same job produces the same result rather than duplicating data.
Essential because pipelines *will* be retried after failures. Achieved with
deterministic partition overwrites, MERGE on a business key, or transactional
commits — which is precisely what an Iceberg snapshot commit gives you.

---

# Part 13 — Traps and weak points, with honest answers

These are the questions that can hurt. Prepared answers below.

### 13.1 "Your tables were only 4–5 MB. Is that realistic?"

**Don't defend it — own it.**

> "No, and it's the main limitation. I made that trade deliberately: each episode
> executes real Spark queries, and I needed 216 evaluation runs plus multi-seed
> training, so an episode had to take minutes rather than hours. The consequence
> is that at that scale, latency is dominated by per-file overhead rather than
> bytes scanned, so my latency/file-count relationship is probably *steeper* than
> production. What I'd claim from it is the *mechanism* — that partitioning
> doesn't pay off without compaction — not the specific magnitudes."

### 13.2 "You said pruning was from a lookup table. So you didn't measure the actual benefit?"

> "Correct, and that's the sharpest criticism of the work. The pruning ratios are
> design assumptions based on semantic alignment between predicate and partition
> key — 92% for a time-range query on an hour-partitioned table, and so on. What I
> did instead was a sensitivity analysis: I scaled every pruning value by 0.8 to
> 1.2 and re-ran. The advantage held at every scale. And notably the winning agent
> actually achieved *lower* pruning than the best heuristic — it won on compaction
> quality, fewer files and lower latency — so the result isn't just the agent
> exploiting the assumed table. But the correct fix is instrumenting Spark's scan
> metrics to count files skipped, and I'd do that first if I continued."

### 13.3 "Why not just use Databricks auto-optimize or Snowflake auto-clustering?"

> "For a static workload, you probably should — and honestly a simple threshold
> heuristic was cheaper than my system. The gap I was targeting is that those
> systems are threshold- and statistics-driven, and they typically maintain file
> compaction and clustering as *separate* subsystems with independent triggers.
> My ablation showed those two decisions interact non-additively — partitioning
> without compaction gets you almost no pruning benefit. That's the thing an
> independent-trigger design can't exploit. I should be clear that's an
> architectural argument, not a benchmark — I couldn't compare directly because
> those optimisers are proprietary and inseparable from their engines."

### 13.4 "You said RL wasn't the cheapest. So why use it?"

> "Right — `Threshold10_C128` cost $8.37 per episode versus $8.79 for the agent,
> because it simply compacted less often. RL buys *reward* — better pruning,
> lower query latency, and robustness when the workload shifts — not minimum
> dollar cost. My honest recommendation in the paper is that for a static workload
> a well-chosen heuristic is the right answer. The learned policy earns its
> complexity when the query mix actually moves, which is where I tested it on
> held-out drifting and mixed workloads."

### 13.5 "Did anything go wrong?" / "What was the hardest part?"

**This is your best answer. Use it.**

> "Two bugs in my own training pipeline that invalidated my first set of results.
> The first was a train/serve normalisation skew — I standardised features by
> per-dataset z-scores during training but used fixed constants at inference, so
> the deployed model was seeing inputs 10–50× outside its training range and had
> effectively collapsed to a constant policy. The apparent adaptivity was an
> artifact.
>
> Fixing that exposed a second, more interesting one: **trajectory-identity
> leakage**. Two of my features were 'steps since last compaction' and 'steps
> since last partition change.' Because I trained offline on logs pooled from
> multiple heuristic policies, those counters effectively encoded *which policy
> generated the trajectory* — near zero for always-compact, unbounded for
> no-maintenance. So the model learned to identify the source policy and clone it,
> rather than read the table state. I caught it with a counterfactual probe: hold
> the table state fixed, vary only that counter, and the probability of 'do
> nothing' went from 0 to 0.94.
>
> I retrained everything and reported both defects in the paper. The lesson I took
> is that no aggregate validation metric would have caught either — I needed a
> targeted probe of a specific hypothesis."

This answer demonstrates debugging skill, statistical care, and intellectual
honesty simultaneously. It's worth more than any result.

### 13.6 "How do you know the improvement is real and not luck?"

> "That's exactly what a reviewer asked. I trained five independent seeds per
> architecture and evaluated each on matched environment seeds, so every policy
> saw identical episodes and the comparisons are paired. I report bootstrap 95%
> confidence intervals, paired permutation and Wilcoxon signed-rank tests, Cliff's
> delta for effect size, and Holm–Bonferroni correction. The headline result is
> p = 0.001 with a large effect, and all five seeds individually beat the best
> heuristic.
>
> The more interesting finding was that one architecture's across-seed standard
> deviation was 0.0403 — three of five seeds beat the heuristic and two failed
> badly. So a single-checkpoint evaluation could have reported almost any ranking.
> That's a methodological result in its own right."

### 13.7 "Isn't this over-engineered? Why not a simple rule?"

Agree partially, then differentiate — never get defensive:

> "For a stationary workload, yes. My own numbers say a threshold heuristic is
> cheaper and simpler, and I say so in the paper. The case for learning is
> specifically non-stationarity plus the coupling between the two decisions. And
> notably the *when to act* tier of my system is still a rule — I tried replacing
> it with a greedy oracle and it performed far worse, because the per-step reward
> difference between choices was smaller than the environment's own measurement
> noise. So part of my conclusion is that rules are the right tool for that layer."

### 13.8 Questions you should NOT invite

Avoid volunteering: specific TensorFlow APIs, PPO's clipped surrogate objective
maths, GAE, entropy regularisation, Transformer attention maths. If he goes
there, give the one-line definition from Part 10 and bridge back:

> "…that's the optimisation detail. What mattered more for the outcome was the
> data side — the feature leakage and the compaction/partition coupling."

---

# Part 14 — Questions you ask him

Have 3 ready. It signals genuine interest.

1. "Does the Data Management track cover open table formats and lakehouse
   architecture, or is it focused on traditional warehousing and SQL?"
2. "How much of the track is hands-on project work versus coursework?"
3. "Which tools do the graduates of this track typically end up working with in
   the Egyptian market — is it more Microsoft/Oracle stack, or cloud-native?"
4. "Is there a capstone project, and do students choose the topic?"

---

# Part 15 — Rapid-fire drill

Cover the answer, say it aloud in one sentence. If you can't, go back to that
part.

| # | Question |
|---|---|
| 1 | Define a lakehouse in one sentence. |
| 2 | Three things a lakehouse gives you that a plain data lake doesn't. |
| 3 | Name Iceberg's four metadata layers, top to bottom. |
| 4 | How does Iceberg achieve atomic commits on S3? |
| 5 | What does "engine-agnostic" actually mean? |
| 6 | What is hidden partitioning and what problem does it solve? |
| 7 | Why is partition evolution possible in Iceberg but not Hive? |
| 8 | Give four distinct reasons small files hurt performance. |
| 9 | Bin-pack vs sort vs Z-order compaction. |
| 10 | Why does a sorted rewrite improve pruning? |
| 11 | Copy-on-write vs merge-on-read — when each? |
| 12 | Describe Parquet's internal layout. |
| 13 | Why is the Parquet footer at the end of the file? |
| 14 | When does predicate pushdown fail to help? |
| 15 | Why does object storage lack atomic rename, and why does that matter? |
| 16 | What does a catalog do in Iceberg? |
| 17 | Narrow vs wide transformation in Spark. |
| 18 | What is data skew and two fixes. |
| 19 | What is AQE and why is it needed on a lake? |
| 20 | Star vs snowflake schema. |
| 21 | SCD Type 1 vs Type 2. |
| 22 | ETL vs ELT and why ELT won. |
| 23 | What is CDC and why is log-based better than polling? |
| 24 | State CAP correctly. |
| 25 | Why did you automate lakehouse maintenance? |
| 26 | What was your single most interesting result? *(the ablation)* |
| 27 | What was your biggest mistake? *(trajectory-identity leakage)* |
| 28 | What's the main limitation of your work? *(scale + assumed pruning)* |
| 29 | Would you recommend RL for this in production? *(conditionally — no for
      static workloads)* |
| 30 | What would you do next? |

---

## Final advice

1. **Lead with data management, not AI.** Your project genuinely is a storage
   problem. Frame it that way.
2. **The ablation finding is your strongest card.** Compaction is a prerequisite
   for partitioning to pay off — measured, counter-intuitive, and yours.
3. **The bug story is your second strongest.** Nobody expects a candidate to
   volunteer that they invalidated their own results and retrained everything.
4. **Never bluff.** His method is depth-first; a bluff fails on the second
   follow-up. "I don't know that part, but here's what I do know" is a
   *good* answer, and it's rarer than you'd think.
5. **Quote real numbers.** +8.3%, p = 0.001, 5 seeds, 216 runs, 0.080 vs 0.225
   pruning, $8.37 vs $8.79. Specificity reads as having actually done the work.
