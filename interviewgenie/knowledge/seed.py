"""Build the seed knowledge graph.

The graph is InterviewGenie's world model.  It is seeded with several hundred
entities and a thousand-plus weighted relationships spanning

* **skills** -- languages, frameworks, databases, cloud platforms, practices,
* **concepts** -- the theory that shows up in interviews,
* **roles** and **companies**,
* **competencies** and **metrics** used to evaluate answers, and
* **behavioural anchors** that turn a STAR story into a scorable claim.

Relations are typed and weighted; the weights are the prior probabilities the
retriever and the continuous-learning module adjust at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..logging import get_logger
from .graph import PropertyGraph

LOG = get_logger("knowledge.seed")


# --------------------------------------------------------------------------- #
# Entity tables
# --------------------------------------------------------------------------- #
SKILLS: Dict[str, Tuple[str, str, List[str]]] = {
    # name: (category, description, aliases)
    "Python": ("language", "Dynamic, batteries-included language dominant in ML and backend services.",
               ["python3", "py"]),
    "Java": ("language", "Statically typed JVM language with a mature concurrency model.", ["jvm"]),
    "JavaScript": ("language", "The language of the browser; event-loop concurrency.", ["js", "ecmascript"]),
    "TypeScript": ("language", "JavaScript with a structural type system.", ["ts"]),
    "Go": ("language", "Compiled language built for networked services and simple concurrency.", ["golang"]),
    "Rust": ("language", "Systems language with ownership-based memory safety and no GC.", ["rustlang"]),
    "C++": ("language", "Systems language with manual memory management and zero-cost abstractions.", ["cpp", "c plus plus"]),
    "C": ("language", "Portable systems language close to the metal.", []),
    "Ruby": ("language", "Developer-happiness-focused dynamic language.", []),
    "Kotlin": ("language", "Modern JVM language with null safety and coroutines.", []),
    "Swift": ("language", "Apple's language for iOS and macOS.", []),
    "Scala": ("language", "JVM language blending OO and functional programming.", []),
    "SQL": ("language", "Declarative language for relational data.", ["structured query language"]),

    "React": ("framework", "Component library for declarative browser UIs.", ["reactjs"]),
    "Vue": ("framework", "Progressive front-end framework.", ["vuejs"]),
    "Angular": ("framework", "Opinionated front-end framework with dependency injection.", ["angularjs"]),
    "Node.js": ("framework", "JavaScript runtime outside the browser.", ["node", "nodejs"]),
    "Django": ("framework", "Batteries-included Python web framework.", []),
    "Flask": ("framework", "Minimal Python web framework.", []),
    "FastAPI": ("framework", "Modern async Python API framework with type-driven validation.", []),
    "Spring Boot": ("framework", "Convention-over-configuration JVM application framework.", ["spring"]),
    "Rails": ("framework", "Convention-over-configuration Ruby web framework.", ["ruby on rails"]),
    ".NET": ("framework", "Microsoft's cross-platform application framework.", ["dotnet", "asp.net"]),

    "TensorFlow": ("ml", "Graph-based deep learning framework with production serving.", ["tf"]),
    "PyTorch": ("ml", "Dynamic-graph deep learning framework dominant in research.", ["torch"]),
    "scikit-learn": ("ml", "Classical machine learning library.", ["sklearn", "scikit learn"]),
    "Pandas": ("data", "DataFrame library for tabular data.", []),
    "NumPy": ("data", "N-dimensional arrays and vectorised maths.", ["numpy"]),
    "Spark": ("data", "Distributed data processing engine.", ["apache spark"]),
    "Kafka": ("infrastructure", "Distributed append-only log for event streaming.", ["apache kafka"]),
    "RabbitMQ": ("infrastructure", "AMQP message broker.", ["rabbit"]),
    "Airflow": ("data", "Workflow orchestration for batch pipelines.", ["apache airflow"]),
    "dbt": ("data", "SQL-first transformation framework for warehouses.", []),

    "PostgreSQL": ("database", "Advanced open-source relational database.", ["postgres", "psql"]),
    "MySQL": ("database", "Widely deployed relational database.", ["mariadb"]),
    "SQLite": ("database", "Embedded relational database.", []),
    "Redis": ("database", "In-memory key-value store used for caching and rate limiting.", []),
    "MongoDB": ("database", "Document database with flexible schemas.", ["mongo"]),
    "Cassandra": ("database", "Wide-column store optimised for write-heavy workloads.", []),
    "Elasticsearch": ("database", "Inverted-index search and analytics engine.", ["elastic"]),
    "Neo4j": ("database", "Native property-graph database.", []),
    "ClickHouse": ("database", "Columnar OLAP database.", []),
    "DynamoDB": ("database", "Managed key-value and document store.", []),

    "Docker": ("infrastructure", "OS-level containerisation.", ["containers"]),
    "Kubernetes": ("infrastructure", "Container orchestration and scheduling.", ["k8s", "kube"]),
    "Terraform": ("infrastructure", "Declarative infrastructure as code.", ["tf"]),
    "AWS": ("cloud", "Amazon Web Services: the broadest cloud platform.", ["amazon web services"]),
    "GCP": ("cloud", "Google Cloud Platform, strong in data and ML.", ["google cloud"]),
    "Azure": ("cloud", "Microsoft's cloud platform.", ["microsoft azure"]),
    "Git": ("practice", "Distributed version control.", []),
    "Linux": ("infrastructure", "The dominant server operating system.", ["unix"]),

    "System design": ("practice", "Translating requirements into a scalable architecture.", ["architecture"]),
    "Distributed systems": ("concept", "Multiple computers cooperating as one system.", ["distribution"]),
    "Data structures": ("concept", "Organised collections that trade time and space.", []),
    "Algorithms": ("concept", "Step-by-step procedures with provable complexity.", []),
    "Machine learning": ("concept", "Systems that improve from data rather than explicit rules.", ["ml"]),
    "Deep learning": ("concept", "Neural networks with many layers.", ["dl"]),
    "Natural language processing": ("concept", "Computational treatment of human language.", ["nlp"]),
    "Computer vision": ("concept", "Extracting meaning from images and video.", ["cv"]),
    "Testing": ("practice", "Automated verification of behaviour.", ["unit testing", "tests"]),
    "CI/CD": ("practice", "Continuous integration and continuous delivery.", ["continuous integration"]),
    "Observability": ("practice", "Metrics, logs and traces that explain system behaviour.", ["monitoring"]),
    "Security": ("practice", "Protecting confidentiality, integrity and availability.", []),
    "Code review": ("practice", "Peer inspection of changes before merge.", []),
    "Mentoring": ("soft", "Growing other engineers deliberately.", ["mentorship"]),
    "Communication": ("soft", "Transferring ideas precisely and concisely.", []),
    "Stakeholder management": ("soft", "Aligning people with different incentives.", []),
    "Conflict resolution": ("soft", "Turning disagreement into a decision.", []),
    "Prioritisation": ("soft", "Choosing what not to do.", ["prioritization"]),
    "Ownership": ("soft", "Carrying an outcome end to end.", []),
    "Incident response": ("practice", "Detecting, mitigating and learning from failures.", ["on call"]),
    "Performance tuning": ("practice", "Finding and removing bottlenecks.", ["optimisation"]),
}

ROLES: Dict[str, Tuple[str, str]] = {
    "Software Engineer": ("ic", "Builds and maintains production software."),
    "Senior Software Engineer": ("ic", "Owns complex features and mentors peers."),
    "Staff Engineer": ("ic", "Sets technical direction across several teams."),
    "Principal Engineer": ("ic", "Company-wide technical leadership."),
    "Engineering Manager": ("management", "Leads people and delivery."),
    "Site Reliability Engineer": ("ic", "Keeps production fast, available and observable."),
    "Data Engineer": ("ic", "Builds pipelines and warehouses."),
    "Machine Learning Engineer": ("ic", "Ships models into production."),
    "Data Scientist": ("ic", "Answers questions with statistics and models."),
    "Product Manager": ("partner", "Decides what gets built and why."),
    "Designer": ("partner", "Shapes the user experience."),
    "DevOps Engineer": ("ic", "Bridges development and operations."),
    "QA Engineer": ("ic", "Owns quality strategy and automation."),
    "Solutions Architect": ("ic", "Designs customer-facing architectures."),
}

COMPANIES: Dict[str, Tuple[str, str]] = {
    "Google": ("bigtech", "Search, ads, cloud and AI."),
    "Meta": ("bigtech", "Social platforms and infrastructure."),
    "Amazon": ("bigtech", "Retail and AWS."),
    "Microsoft": ("bigtech", "Software, cloud and developer tools."),
    "Apple": ("bigtech", "Consumer hardware and services."),
    "Netflix": ("bigtech", "Streaming at extreme scale."),
    "Stripe": ("scaleup", "Payments infrastructure."),
    "Airbnb": ("scaleup", "Travel marketplace."),
    "Uber": ("scaleup", "Mobility and delivery."),
    "Spotify": ("scaleup", "Music streaming."),
    "Shopify": ("scaleup", "E-commerce platform."),
    "Databricks": ("scaleup", "Lakehouse data platform."),
    "Snowflake": ("scaleup", "Cloud data warehouse."),
    "Figma": ("scaleup", "Collaborative design tool."),
    "Canva": ("scaleup", "Visual communication platform."),
    "Atlassian": ("scaleup", "Developer collaboration tools."),
}

CONCEPTS: Dict[str, str] = {
    "CAP theorem": "A distributed system can provide at most two of consistency, availability and partition tolerance.",
    "ACID": "Atomicity, consistency, isolation and durability for transactions.",
    "BASE": "Basically available, soft state, eventual consistency.",
    "Eventual consistency": "Replicas converge if updates stop and reads are retried.",
    "Strong consistency": "Every read sees the latest write.",
    "Consensus": "Getting independent nodes to agree on one value.",
    "Paxos": "Classic consensus protocol family.",
    "Raft": "Consensus protocol designed for understandability.",
    "Sharding": "Splitting a dataset across nodes by key.",
    "Replication": "Keeping copies of data on multiple nodes.",
    "Load balancing": "Spreading requests across capacity.",
    "Caching": "Storing expensive results closer to the reader.",
    "Rate limiting": "Bounding request rates to protect a service.",
    "Backpressure": "Slowing producers when consumers fall behind.",
    "Idempotency": "Repeating an operation has the same effect as doing it once.",
    "Circuit breaker": "Failing fast when a dependency is unhealthy.",
    "Queue": "Decoupling producers and consumers with a buffer.",
    "Event-driven architecture": "Components react to events rather than calls.",
    "Microservices": "Independently deployable services with narrow scope.",
    "Monolith": "A single deployable unit containing all behaviour.",
    "Service mesh": "Infrastructure layer handling service-to-service traffic.",
    "Big O notation": "Asymptotic growth of time or space with input size.",
    "Hash table": "Key-value structure with expected O(1) lookup.",
    "B-tree": "Balanced tree optimised for disk-based range reads.",
    "LSM tree": "Write-optimised structure that merges sorted runs.",
    "Index": "Auxiliary structure that accelerates lookups.",
    "Normalisation": "Organising tables to remove redundancy.",
    "Transaction isolation": "How concurrent transactions observe each other.",
    "Deadlock": "Circular wait where no participant can proceed.",
    "Garbage collection": "Automatic reclamation of unreachable memory.",
    "Concurrency": "Making progress on multiple tasks at once.",
    "Parallelism": "Executing multiple tasks simultaneously.",
    "Race condition": "Outcome depends on non-deterministic interleaving.",
    "Mutex": "Mutual-exclusion lock.",
    "Semaphore": "Counter bounding concurrent access.",
    "TCP": "Reliable, ordered, connection-oriented transport.",
    "UDP": "Unreliable, unordered, connectionless transport.",
    "HTTP": "Request-response application protocol.",
    "HTTPS": "HTTP over TLS.",
    "TLS": "Transport encryption with authentication.",
    "REST": "Resource-oriented HTTP API style.",
    "GraphQL": "Query language letting clients pick fields.",
    "gRPC": "High-performance RPC over HTTP/2 with protobuf.",
    "Authentication": "Proving who you are.",
    "Authorisation": "Deciding what you may do.",
    "Encryption at rest": "Protecting stored data from disclosure.",
    "Principle of least privilege": "Granting the minimum access required.",
    "Technical debt": "Deliberate or accidental shortcuts that slow future work.",
    "Refactoring": "Restructuring code without changing behaviour.",
    "Test-driven development": "Write a failing test, then make it pass.",
    "Continuous deployment": "Every merged change reaches production automatically.",
    "Feature flag": "Runtime switch gating new behaviour.",
    "Canary release": "Exposing a change to a small traffic slice first.",
    "Blue-green deployment": "Switching traffic between two identical environments.",
    "Postmortem": "Blameless analysis of an incident.",
    "Error budget": "Allowed unreliability that balances velocity and stability.",
    "Service level objective": "Target reliability for a service.",
    "Latency": "Time to complete one operation.",
    "Throughput": "Operations completed per unit time.",
    "Availability": "Fraction of time a service answers correctly.",
    "p99 latency": "Latency below which 99% of requests complete.",
    "A/B test": "Randomised comparison of two variants.",
    "Feature engineering": "Turning raw data into model inputs.",
    "Overfitting": "Model memorises noise instead of the signal.",
    "Bias-variance tradeoff": "Error from wrong assumptions versus sensitivity to data.",
    "Embedding": "Dense vector representation of an entity.",
    "Transformer": "Attention-based architecture behind modern language models.",
    "Retrieval-augmented generation": "Grounding generation in retrieved documents.",
    "STAR method": "Situation, task, action, result for behavioural answers.",
    "Psychological safety": "Belief that interpersonal risk-taking is safe.",
}

#: behavioural competency anchors used to score answers
COMPETENCIES: Dict[str, str] = {
    "Problem solving": "Decomposing ambiguity into tractable steps.",
    "Collaboration": "Working effectively with people who think differently.",
    "Communication": "Explaining complex ideas to varied audiences.",
    "Ownership": "Seeing an outcome through without being asked.",
    "Adaptability": "Staying effective when the situation changes.",
    "Mentorship": "Raising the level of the people around you.",
    "Decision making": "Choosing well with incomplete information.",
    "Influence without authority": "Driving change through persuasion.",
    "Resilience": "Recovering from setbacks without losing momentum.",
    "Craft": "Care about the quality of the thing you ship.",
}

METRICS: Dict[str, str] = {
    "Deployment frequency": "How often changes reach production.",
    "Lead time for changes": "Time from commit to production.",
    "Change failure rate": "Fraction of changes causing an incident.",
    "Mean time to recovery": "Time to restore service after an incident.",
    "Test coverage": "Fraction of code exercised by tests.",
    "Code review turnaround": "Time for a change to be reviewed.",
    "Customer satisfaction": "How happy users are with the product.",
    "Error rate": "Fraction of requests that fail.",
    "P95 latency": "Latency below which 95% of requests complete.",
    "Cost per request": "Infrastructure spend divided by traffic.",
}

#: relations seeded between entity families: (a, relation, b, weight)
_RELATIONS: List[Tuple[str, str, str, float]] = [
    # languages -> frameworks / runtimes
    ("Python", "used_by", "Django", 0.9),
    ("Python", "used_by", "Flask", 0.9),
    ("Python", "used_by", "FastAPI", 0.95),
    ("Python", "used_by", "TensorFlow", 0.85),
    ("Python", "used_by", "PyTorch", 0.9),
    ("Python", "used_by", "scikit-learn", 0.95),
    ("Python", "used_by", "Pandas", 0.95),
    ("Python", "used_by", "NumPy", 0.95),
    ("Python", "used_by", "Airflow", 0.85),
    ("Java", "used_by", "Spring Boot", 0.95),
    ("Java", "used_by", "Kafka", 0.85),
    ("Java", "used_by", "Spark", 0.8),
    ("Java", "used_by", "Cassandra", 0.8),
    ("JavaScript", "used_by", "React", 0.95),
    ("JavaScript", "used_by", "Vue", 0.95),
    ("JavaScript", "used_by", "Angular", 0.95),
    ("JavaScript", "used_by", "Node.js", 0.95),
    ("TypeScript", "used_by", "React", 0.85),
    ("TypeScript", "used_by", "Angular", 0.85),
    ("TypeScript", "used_by", "Node.js", 0.8),
    ("Go", "used_by", "Docker", 0.95),
    ("Go", "used_by", "Kubernetes", 0.95),
    ("Go", "used_by", "Terraform", 0.9),
    ("Go", "used_by", "Kafka", 0.6),
    ("Rust", "used_by", "Terraform", 0.4),
    ("SQL", "used_by", "PostgreSQL", 0.95),
    ("SQL", "used_by", "MySQL", 0.95),
    ("SQL", "used_by", "SQLite", 0.95),
    ("SQL", "used_by", "ClickHouse", 0.9),
    ("SQL", "used_by", "dbt", 0.95),

    # concepts -> practice
    ("CAP theorem", "related_to", "Distributed systems", 0.95),
    ("CAP theorem", "related_to", "Eventual consistency", 0.9),
    ("CAP theorem", "related_to", "Strong consistency", 0.85),
    ("ACID", "related_to", "PostgreSQL", 0.85),
    ("ACID", "related_to", "Transaction isolation", 0.95),
    ("BASE", "related_to", "Eventual consistency", 0.95),
    ("Consensus", "related_to", "Paxos", 0.95),
    ("Consensus", "related_to", "Raft", 0.95),
    ("Consensus", "related_to", "Distributed systems", 0.9),
    ("Sharding", "related_to", "MongoDB", 0.8),
    ("Sharding", "related_to", "Cassandra", 0.9),
    ("Sharding", "related_to", "DynamoDB", 0.8),
    ("Replication", "related_to", "PostgreSQL", 0.85),
    ("Replication", "related_to", "Kafka", 0.75),
    ("Load balancing", "related_to", "AWS", 0.7),
    ("Caching", "related_to", "Redis", 0.95),
    ("Caching", "related_to", "Latency", 0.85),
    ("Rate limiting", "related_to", "Redis", 0.85),
    ("Rate limiting", "related_to", "Throughput", 0.7),
    ("Backpressure", "related_to", "Queue", 0.85),
    ("Idempotency", "related_to", "Queue", 0.8),
    ("Circuit breaker", "related_to", "Microservices", 0.85),
    ("Circuit breaker", "related_to", "Availability", 0.8),
    ("Queue", "related_to", "Kafka", 0.9),
    ("Queue", "related_to", "RabbitMQ", 0.9),
    ("Event-driven architecture", "related_to", "Kafka", 0.9),
    ("Event-driven architecture", "related_to", "Microservices", 0.85),
    ("Microservices", "contrasts_with", "Monolith", 0.9),
    ("Microservices", "related_to", "Service mesh", 0.8),
    ("Microservices", "related_to", "Kubernetes", 0.85),
    ("Monolith", "related_to", "Technical debt", 0.6),
    ("Big O notation", "related_to", "Algorithms", 0.95),
    ("Big O notation", "related_to", "Data structures", 0.9),
    ("Hash table", "related_to", "Data structures", 0.95),
    ("Hash table", "related_to", "Redis", 0.7),
    ("B-tree", "related_to", "Index", 0.95),
    ("B-tree", "related_to", "PostgreSQL", 0.85),
    ("LSM tree", "related_to", "Cassandra", 0.9),
    ("LSM tree", "related_to", "Kafka", 0.7),
    ("Index", "related_to", "Elasticsearch", 0.85),
    ("Normalisation", "related_to", "SQL", 0.8),
    ("Transaction isolation", "related_to", "Deadlock", 0.7),
    ("Deadlock", "related_to", "Mutex", 0.85),
    ("Deadlock", "related_to", "Concurrency", 0.85),
    ("Garbage collection", "related_to", "Java", 0.85),
    ("Concurrency", "related_to", "Parallelism", 0.9),
    ("Concurrency", "related_to", "Race condition", 0.9),
    ("Race condition", "related_to", "Testing", 0.6),
    ("Mutex", "related_to", "Semaphore", 0.85),
    ("TCP", "related_to", "UDP", 0.9),
    ("TCP", "related_to", "HTTP", 0.85),
    ("UDP", "related_to", "gRPC", 0.5),
    ("HTTP", "related_to", "REST", 0.95),
    ("HTTP", "related_to", "GraphQL", 0.85),
    ("HTTPS", "related_to", "TLS", 0.95),
    ("REST", "contrasts_with", "GraphQL", 0.85),
    ("REST", "related_to", "gRPC", 0.8),
    ("Authentication", "related_to", "Authorisation", 0.95),
    ("Authorisation", "related_to", "Principle of least privilege", 0.9),
    ("Technical debt", "related_to", "Refactoring", 0.9),
    ("Refactoring", "related_to", "Testing", 0.85),
    ("Test-driven development", "related_to", "Testing", 0.95),
    ("Continuous deployment", "related_to", "CI/CD", 0.95),
    ("Continuous deployment", "related_to", "Canary release", 0.9),
    ("Continuous deployment", "related_to", "Feature flag", 0.9),
    ("Canary release", "related_to", "Blue-green deployment", 0.85),
    ("Postmortem", "related_to", "Incident response", 0.95),
    ("Error budget", "related_to", "Service level objective", 0.95),
    ("Service level objective", "related_to", "Availability", 0.9),
    ("Latency", "related_to", "p99 latency", 0.95),
    ("Throughput", "related_to", "Performance tuning", 0.85),
    ("Availability", "related_to", "Observability", 0.8),
    ("A/B test", "related_to", "Machine learning", 0.6),
    ("Feature engineering", "related_to", "Machine learning", 0.95),
    ("Overfitting", "related_to", "Bias-variance tradeoff", 0.95),
    ("Embedding", "related_to", "Natural language processing", 0.85),
    ("Transformer", "related_to", "Deep learning", 0.95),
    ("Retrieval-augmented generation", "related_to", "Transformer", 0.9),
    ("Retrieval-augmented generation", "related_to", "Embedding", 0.85),
    ("STAR method", "related_to", "Communication", 0.85),
    ("STAR method", "related_to", "Ownership", 0.7),

    # observability & SRE
    ("Observability", "related_to", "Incident response", 0.9),
    ("Observability", "related_to", "Postmortem", 0.85),
    ("Incident response", "measured_by", "Mean time to recovery", 0.9),
    ("Incident response", "measured_by", "Change failure rate", 0.8),
    ("CI/CD", "measured_by", "Deployment frequency", 0.9),
    ("CI/CD", "measured_by", "Lead time for changes", 0.9),
    ("Testing", "measured_by", "Test coverage", 0.9),
    ("Testing", "measured_by", "Change failure rate", 0.7),
    ("Code review", "measured_by", "Code review turnaround", 0.9),
    ("Performance tuning", "measured_by", "P95 latency", 0.85),
    ("Performance tuning", "measured_by", "Throughput", 0.8),
    ("Microservices", "measured_by", "Error rate", 0.7),
    ("Machine learning", "measured_by", "P95 latency", 0.4),

    # roles -> skills
    ("Software Engineer", "requires", "Data structures", 0.9),
    ("Software Engineer", "requires", "Algorithms", 0.9),
    ("Software Engineer", "requires", "Git", 0.95),
    ("Software Engineer", "requires", "Testing", 0.9),
    ("Software Engineer", "requires", "Communication", 0.85),
    ("Senior Software Engineer", "requires", "System design", 0.95),
    ("Senior Software Engineer", "requires", "Mentoring", 0.85),
    ("Senior Software Engineer", "requires", "Code review", 0.9),
    ("Staff Engineer", "requires", "Distributed systems", 0.9),
    ("Staff Engineer", "requires", "Influence without authority", 0.95),
    ("Staff Engineer", "requires", "Technical debt", 0.7),
    ("Engineering Manager", "requires", "Stakeholder management", 0.95),
    ("Engineering Manager", "requires", "Mentoring", 0.95),
    ("Engineering Manager", "requires", "Prioritisation", 0.95),
    ("Site Reliability Engineer", "requires", "Observability", 0.95),
    ("Site Reliability Engineer", "requires", "Incident response", 0.95),
    ("Site Reliability Engineer", "requires", "Linux", 0.9),
    ("Site Reliability Engineer", "requires", "Kubernetes", 0.85),
    ("Data Engineer", "requires", "SQL", 0.95),
    ("Data Engineer", "requires", "Spark", 0.9),
    ("Data Engineer", "requires", "Kafka", 0.85),
    ("Machine Learning Engineer", "requires", "Machine learning", 0.95),
    ("Machine Learning Engineer", "requires", "Python", 0.9),
    ("Machine Learning Engineer", "requires", "Observability", 0.7),
    ("Data Scientist", "requires", "Machine learning", 0.95),
    ("Data Scientist", "requires", "SQL", 0.85),
    ("DevOps Engineer", "requires", "Docker", 0.95),
    ("DevOps Engineer", "requires", "Kubernetes", 0.9),
    ("DevOps Engineer", "requires", "Terraform", 0.9),
    ("DevOps Engineer", "requires", "CI/CD", 0.95),
    ("Solutions Architect", "requires", "System design", 0.95),
    ("Solutions Architect", "requires", "AWS", 0.85),
    ("Solutions Architect", "requires", "Stakeholder management", 0.85),
    ("Product Manager", "requires", "Prioritisation", 0.95),
    ("Product Manager", "requires", "Communication", 0.95),
    ("Product Manager", "requires", "A/B test", 0.7),

    # competencies -> roles
    ("Problem solving", "demonstrated_in", "Software Engineer", 0.9),
    ("Collaboration", "demonstrated_in", "Software Engineer", 0.85),
    ("Ownership", "demonstrated_in", "Senior Software Engineer", 0.9),
    ("Mentorship", "demonstrated_in", "Senior Software Engineer", 0.85),
    ("Decision making", "demonstrated_in", "Staff Engineer", 0.9),
    ("Influence without authority", "demonstrated_in", "Staff Engineer", 0.95),
    ("Resilience", "demonstrated_in", "Engineering Manager", 0.8),
    ("Adaptability", "demonstrated_in", "Software Engineer", 0.8),
    ("Craft", "demonstrated_in", "Software Engineer", 0.8),

    # companies -> stack
    ("Google", "uses", "Python", 0.8),
    ("Google", "uses", "TensorFlow", 0.9),
    ("Google", "uses", "Kubernetes", 0.95),
    ("Google", "uses", "Distributed systems", 0.95),
    ("Meta", "uses", "Python", 0.7),
    ("Meta", "uses", "PyTorch", 0.9),
    ("Meta", "uses", "Distributed systems", 0.9),
    ("Amazon", "uses", "Java", 0.85),
    ("Amazon", "uses", "AWS", 0.95),
    ("Amazon", "uses", "DynamoDB", 0.8),
    ("Microsoft", "uses", ".NET", 0.9),
    ("Microsoft", "uses", "TypeScript", 0.8),
    ("Microsoft", "uses", "Azure", 0.95),
    ("Netflix", "uses", "Java", 0.8),
    ("Netflix", "uses", "Kafka", 0.9),
    ("Netflix", "uses", "Observability", 0.95),
    ("Stripe", "uses", "Ruby", 0.7),
    ("Stripe", "uses", "Go", 0.7),
    ("Stripe", "uses", "PostgreSQL", 0.8),
    ("Airbnb", "uses", "JavaScript", 0.7),
    ("Airbnb", "uses", "Airflow", 0.8),
    ("Uber", "uses", "Go", 0.8),
    ("Uber", "uses", "Kafka", 0.85),
    ("Spotify", "uses", "Java", 0.7),
    ("Spotify", "uses", "Google", 0.6),
    ("Shopify", "uses", "Ruby", 0.9),
    ("Databricks", "uses", "Spark", 0.95),
    ("Databricks", "uses", "Python", 0.85),
    ("Snowflake", "uses", "SQL", 0.95),
    ("Figma", "uses", "TypeScript", 0.8),
    ("Canva", "uses", "Java", 0.7),
    ("Atlassian", "uses", "Java", 0.7),

    # alternatives
    ("React", "alternative_to", "Vue", 0.8),
    ("React", "alternative_to", "Angular", 0.8),
    ("PostgreSQL", "alternative_to", "MySQL", 0.8),
    ("PostgreSQL", "alternative_to", "MongoDB", 0.6),
    ("Redis", "alternative_to", "Memcached", 0.7),
    ("Kafka", "alternative_to", "RabbitMQ", 0.7),
    ("AWS", "alternative_to", "GCP", 0.8),
    ("AWS", "alternative_to", "Azure", 0.8),
    ("PyTorch", "alternative_to", "TensorFlow", 0.85),
    ("Kubernetes", "alternative_to", "Nomad", 0.6),
    ("REST", "alternative_to", "gRPC", 0.7),
    ("Docker", "alternative_to", "Podman", 0.6),

    # prerequisites
    ("Distributed systems", "prerequisite_of", "System design", 0.9),
    ("Data structures", "prerequisite_of", "Algorithms", 0.9),
    ("Algorithms", "prerequisite_of", "Machine learning", 0.8),
    ("SQL", "prerequisite_of", "Data Engineer", 0.85),
    ("Testing", "prerequisite_of", "CI/CD", 0.85),
    ("Git", "prerequisite_of", "CI/CD", 0.8),
    ("Linux", "prerequisite_of", "Observability", 0.7),
    ("Machine learning", "prerequisite_of", "Deep learning", 0.9),
    ("Deep learning", "prerequisite_of", "Natural language processing", 0.85),
    ("Communication", "prerequisite_of", "Stakeholder management", 0.8),
    ("Mentoring", "prerequisite_of", "Engineering Manager", 0.75),

    # connect the tool families to the underlying theory
    ("PyTorch", "implements", "Deep learning", 0.95),
    ("TensorFlow", "implements", "Deep learning", 0.95),
    ("scikit-learn", "implements", "Machine learning", 0.95),
    ("Deep learning", "related_to", "Transformer", 0.9),
    ("Machine learning", "related_to", "Overfitting", 0.9),
    ("Machine learning", "related_to", "Feature engineering", 0.9),
    ("Kafka", "implements", "Queue", 0.95),
    ("RabbitMQ", "implements", "Queue", 0.95),
    ("Redis", "implements", "Caching", 0.95),
    ("Kubernetes", "implements", "Load balancing", 0.6),
    ("PostgreSQL", "implements", "ACID", 0.85),
    ("PostgreSQL", "implements", "Index", 0.85),
    ("MongoDB", "implements", "Sharding", 0.8),
    ("Cassandra", "implements", "LSM tree", 0.9),
    ("Elasticsearch", "implements", "Index", 0.9),
    ("Docker", "implements", "Microservices", 0.6),
    ("Git", "implements", "CI/CD", 0.8),
    ("Terraform", "implements", "CI/CD", 0.7),
    ("Security", "related_to", "Encryption at rest", 0.9),
    ("Security", "related_to", "Authentication", 0.9),
    ("Security", "related_to", "Principle of least privilege", 0.9),
    ("Security", "related_to", "TLS", 0.8),
    ("Observability", "related_to", "Latency", 0.85),
    ("Observability", "related_to", "Throughput", 0.8),
    ("System design", "related_to", "Latency", 0.85),
    ("System design", "related_to", "Throughput", 0.85),
    ("System design", "related_to", "Availability", 0.85),
    ("System design", "related_to", "Sharding", 0.85),
    ("System design", "related_to", "Caching", 0.85),
    ("System design", "related_to", "Rate limiting", 0.8),
    ("System design", "related_to", "Load balancing", 0.85),
    ("System design", "related_to", "Microservices", 0.8),
    ("Python", "used_by", "dbt", 0.7),
    ("Python", "related_to", "Data structures", 0.7),
    ("Java", "related_to", "Garbage collection", 0.9),
    ("Java", "related_to", "Concurrency", 0.85),
    ("Go", "related_to", "Concurrency", 0.85),
    ("Rust", "related_to", "Concurrency", 0.7),
    ("JavaScript", "related_to", "Concurrency", 0.8),
    ("Natural language processing", "related_to", "Machine learning", 0.9),
    ("Natural language processing", "related_to", "Retrieval-augmented generation", 0.9),
    ("Computer vision", "related_to", "Deep learning", 0.9),
    ("Algorithms", "related_to", "Big O notation", 0.9),
    ("Data structures", "related_to", "Hash table", 0.9),
    ("Data structures", "related_to", "B-tree", 0.85),
    ("Communication", "related_to", "STAR method", 0.8),
    ("Collaboration", "related_to", "Conflict resolution", 0.9),
    ("Collaboration", "related_to", "Psychological safety", 0.8),
    ("Prioritisation", "related_to", "Technical debt", 0.75),
    ("Ownership", "related_to", "Incident response", 0.7),
    ("Adaptability", "related_to", "Psychological safety", 0.7),
    ("Resilience", "related_to", "Postmortem", 0.7),
    ("Craft", "related_to", "Code review", 0.8),
    ("Craft", "related_to", "Refactoring", 0.8),
    ("Problem solving", "related_to", "Decision making", 0.85),
    ("Influence without authority", "related_to", "Communication", 0.85),
    ("Site Reliability Engineer", "uses", "Error budget", 0.85),
    ("Engineering Manager", "uses", "Deployment frequency", 0.7),
    ("Data Engineer", "uses", "Airflow", 0.85),
    ("Data Engineer", "uses", "dbt", 0.8),
    ("Machine Learning Engineer", "uses", "PyTorch", 0.85),
    ("Data Scientist", "uses", "scikit-learn", 0.85),
    ("DevOps Engineer", "uses", "AWS", 0.8),
    ("Software Engineer", "uses", "Git", 0.9),
    ("Software Engineer", "uses", "Code review", 0.85),
    ("Senior Software Engineer", "uses", "System design", 0.9),
    ("Staff Engineer", "uses", "Technical debt", 0.8),
    ("Solutions Architect", "uses", "GCP", 0.7),
    ("Product Manager", "uses", "A/B test", 0.8),
]


def build_seed_graph() -> PropertyGraph:
    """Construct the full seed graph."""
    graph = PropertyGraph("interviewgenie")

    for name, (category, description, aliases) in SKILLS.items():
        graph.add_node(f"skill:{name}", {"SKILL", "ENTITY"}, name=name,
                       category=category, description=description)
        for alias in aliases:
            graph.register_alias(alias, f"skill:{name}")

    for name, (level, description) in ROLES.items():
        graph.add_node(f"role:{name}", {"ROLE", "ENTITY"}, name=name,
                       level=level, description=description)

    for name, (size, description) in COMPANIES.items():
        graph.add_node(f"company:{name}", {"COMPANY", "ENTITY"}, name=name,
                       size=size, description=description)

    for name, description in CONCEPTS.items():
        graph.add_node(f"concept:{name}", {"CONCEPT", "ENTITY"}, name=name,
                       category="concept", description=description)

    for name, description in COMPETENCIES.items():
        graph.add_node(f"competency:{name}", {"COMPETENCY", "ENTITY"}, name=name,
                       category="competency", description=description)

    for name, description in METRICS.items():
        graph.add_node(f"metric:{name}", {"METRIC", "ENTITY"}, name=name,
                       category="metric", description=description)

    # extra nodes referenced by relations but not defined above
    _EXTRA_NODES = {
        "Memcached": ("skill", "Distributed in-memory cache.", []),
        "Nomad": ("skill", "HashiCorp's simple scheduler.", []),
        "Podman": ("skill", "Daemonless container engine.", []),
    }
    for name, (category, description, aliases) in _EXTRA_NODES.items():
        node_id = f"skill:{name}"
        if node_id not in graph.nodes:
            graph.add_node(node_id, {"SKILL", "ENTITY"}, name=name,
                           category=category, description=description)
            for alias in aliases:
                graph.register_alias(alias, node_id)

    def resolve(name: str) -> str:
        for prefix in ("skill", "role", "company", "concept", "competency", "metric"):
            node_id = f"{prefix}:{name}"
            if node_id in graph.nodes:
                return node_id
        raise KeyError(name)

    for a, relation, b, weight in _RELATIONS:
        try:
            graph.add_edge(resolve(a), resolve(b), relation, weight)
        except KeyError as exc:
            LOG.debug("skipping relation with unknown entity: %s", exc)
        except Exception as exc:  # noqa: BLE001 - seeding must never crash startup
            LOG.debug("could not add relation %s: %s", (a, relation, b), exc)

    # register canonical names as aliases too
    for node_id, node in list(graph.nodes.items()):
        name = node.properties.get("name")
        if name:
            graph.register_alias(str(name), node_id)

    LOG.info("seed knowledge graph built", context=graph.stats())
    return graph


#: entities used by the entity linker's gazetteer
def alias_index(graph: PropertyGraph) -> Dict[str, Tuple[str, str]]:
    """``alias -> (node_id, label)`` for entity linking."""
    out: Dict[str, Tuple[str, str]] = {}
    for alias, node_id in graph.aliases().items():
        node = graph.get(node_id)
        if node is None:
            continue
        out[alias] = (node_id, node.label() or "ENTITY")
    return out
