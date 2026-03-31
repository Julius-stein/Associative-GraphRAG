# Keyword Drift Case Study

This file collects direct evidence for the keyword-drift pre-experiment. The goal is not only to report Jaccard values, but to show concrete keyword outputs and graph-entry deviations.

## 1. Agriculture: `q002`

Group id: `q002`  
Base query:

`How does the book suggest new beekeepers to maintain their commitment to beekeeping?`

This is a good case because the group-level score is already extreme:

- `low_keyword_jaccard_avg = 0.0211`
- `low_keyword_overlap_at_k_avg = 0.02`

### 1.1 Base query

Query:

`How does the book suggest new beekeepers to maintain their commitment to beekeeping?`

High-level keywords:

- `Beekeeping`
- `Commitment maintenance`
- `New beekeepers`

Low-level keywords:

- `Book`
- `Beekeeping tips`
- `Motivation strategies`
- `Time management`
- `Beginner advice`

Local entry top-8:

- `"BEGINNER BEEKEEPING BOOKS"`
- `"BEEKEEPING: A PRACTICAL GUIDE"`
- `"BEEKEEPING PUBLICATIONS"`
- `"BEEKEEPING MAGAZINES, JOURNALS, AND BOOKS"`
- `"APPENDIX A"`
- `"BEEKEEPING MAGAZINE"`
- `"BEEKEEPING TECHNIQUES"`
- `"BEEKEEPING CATALOG"`

Global entry top-8:

- `"BEGINNING BEEKEEPERS" -> "EXPERIENCED BEEKEEPERS"`
- `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
- `"BEEKEEPER" -> "PACKAGE BEES"`
- `"BEEKEEPER" -> "HIVES"`
- `"BEEKEEPERS' CLUB" -> "BEEKEEPING"`
- `"BEEKEEPER" -> "ESTABLISHED COLONY"`
- `"BEEKEEPING" -> "INDIVIDUALS"`
- `"BEEKEEPING" -> "MILLER"`

Subgraph size:

- `110 entities / 194 relations / 184 sources`

### 1.2 Rewrite 1

Query:

`How does the book advise novice beekeepers to stay dedicated to beekeeping?`

High-level keywords:

- `Beekeeping`
- `Novice guidance`
- `Dedication`
- `Skill development`

Low-level keywords:

- `Book advice`
- `Novice beekeepers`
- `Consistency`
- `Routine management`
- `Hive maintenance`
- `Learning techniques`

Local entry top-8:

- `"BEGINNER BEEKEEPING BOOKS"`
- `"BEEKEEPING TECHNIQUES"`
- `"BEEKEEPING: A PRACTICAL GUIDE"`
- `"EARLY PRACTICES OF BEEKEEPING"`
- `"APPENDIX A"`
- `"NEW BEEKEEPER"`
- `"BEEKEEPING"`
- `"BEGINNING BEEKEEPERS"`

Global entry top-8:

- `"BEGINNING BEEKEEPERS" -> "EXPERIENCED BEEKEEPERS"`
- `"EXPERIENCED APICULTURIST" -> "NOVICE BEEKEEPER"`
- `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
- `"BEE SCHOOL" -> "BEEKEEPING"`
- `"BEES" -> "EXPERIENCED BEEKEEPER"`
- `"BEEKEEPER" -> "PACKAGE BEES"`
- `"BEEKEEPING COMMUNITY" -> "BEGINNER BEEKEEPING BOOKS"`
- `"BEEKEEPER" -> "THE FIRST YEAR"`

Subgraph size:

- `127 entities / 189 relations / 211 sources`

### 1.3 What this shows

Human reading says these two questions are semantically equivalent. But the actual keyword outputs shift hard:

- base low keywords focus on:
  `motivation`, `time management`, `beginner advice`
- rewrite low keywords shift to:
  `consistency`, `routine management`, `hive maintenance`, `learning techniques`

This is not a mild lexical variation. The graph entry also shifts:

- local entry pulls in more operational / technique-oriented nodes like:
  `"EARLY PRACTICES OF BEEKEEPING"`, `"NEW BEEKEEPER"`, `"BEEKEEPING"`
- global entry keeps hitting edges such as:
  `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
  and
  `"BEEKEEPER" -> "PACKAGE BEES"`

These are much more about beginner beekeeping operations than about "how to maintain commitment". This is a direct visual example of keyword drift pushing graph entry away from the actual intent.

### 1.4 Full `q002` trace snapshot

For completeness, the whole `q002` group is included below.

#### `rw2`

Query:

`What methods does the book suggest for new beekeepers to maintain their commitment to beekeeping?`

High-level keywords:

- `Beekeeping`
- `Commitment`
- `Skill development`
- `New beekeepers`

Low-level keywords:

- `Book methods`
- `Maintenance strategies`
- `Practice routines`
- `Motivation techniques`
- `Beekeeping tips`

Local entry top-8:

- `"BEEKEEPING TECHNIQUES"`
- `"BEEKEEPING"`
- `"BEEKEEPING PRACTICES"`
- `"EARLY PRACTICES OF BEEKEEPING"`
- `"BEEKEEPING MAGAZINES, JOURNALS, AND BOOKS"`
- `"BEEKEEPING PUBLICATIONS"`
- `"BEEKEEPING MANUAL"`
- `"APPENDIX A"`

Global entry top-8:

- `"BEGINNING BEEKEEPERS" -> "EXPERIENCED BEEKEEPERS"`
- `"BEE SCHOOL" -> "BEEKEEPING"`
- `"BEES" -> "EXPERIENCED BEEKEEPER"`
- `"EXPERIENCED APICULTURIST" -> "NOVICE BEEKEEPER"`
- `"BEEKEEPING" -> "INDIVIDUALS"`
- `"BEEKEEPER" -> "THE FIRST YEAR"`
- `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
- `"BEEKEEPING" -> "MILLER"`

Subgraph size:

- `123 entities / 170 relations / 131 sources`

#### `rw3`

Query:

`How does the book recommend new beekeepers keep up their regular commitment to beekeeping?`

High-level keywords:

- `Beekeeping`
- `Commitment maintenance`
- `Beginner guidance`

Low-level keywords:

- `Book recommendations`
- `New beekeepers`
- `Regular schedule`
- `Daily/weekly tasks`
- `Hive management`
- `Record keeping`

Local entry top-8:

- `"NEW BEEKEEPER"`
- `"BEGINNER BEEKEEPING BOOKS"`
- `"BEEKEEPING"`
- `"BEEKEEPER"`
- `"BEEKEEPING LIFE"`
- `"BEEKEEPING PUBLICATIONS"`
- `"BEEKEEPING MAGAZINES, JOURNALS, AND BOOKS"`
- `"BEEKEEPERS"`

Global entry top-8:

- `"BEGINNING BEEKEEPERS" -> "EXPERIENCED BEEKEEPERS"`
- `"BEEKEEPING COMMUNITY" -> "BEGINNER BEEKEEPING BOOKS"`
- `"BEEKEEPER" -> "PACKAGE BEES"`
- `"BEES" -> "EXPERIENCED BEEKEEPER"`
- `"BEEKEEPING ORGANIZATIONS" -> "US BEEKEEPERS"`
- `"BEEKEEPING" -> "MILLER"`
- `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
- `"BEEKEEPING" -> "HONEY BEE"`

Subgraph size:

- `128 entities / 164 relations / 101 sources`

#### `rw4`

Query:

`How should new beekeepers stay committed according to the book?`

High-level keywords:

- `Beekeeping`
- `Commitment`
- `New beekeepers`
- `Book insights`

Low-level keywords:

- `Beekeeping practices`
- `Book recommendations`
- `Learning process`
- `Challenges`
- `Guidelines`

Local entry top-8:

- `"BEEKEEPING"`
- `"BEEKEEPING: A PRACTICAL GUIDE"`
- `"BEEKEEPING LIFE"`
- `"BEEKEEPING MANUAL"`
- `"EARLY PRACTICES OF BEEKEEPING"`
- `"BEEKEEPING TECHNIQUES"`
- `"BEEKEEPING EXPERIENCE"`
- `"BEGINNER BEEKEEPING BOOKS"`

Global entry top-8:

- `"BEEKEEPERS" -> "I"`
- `"BEEKEEPING COMMUNITY" -> "BEGINNER BEEKEEPING BOOKS"`
- `"BEGINNING BEEKEEPERS" -> "EXPERIENCED BEEKEEPERS"`
- `"NOVICE BEEKEEPER" -> "SWARM SEASON"`
- `"BEEKEEPING" -> "MILLER"`
- `"BEEKEEPER" -> "THE FIRST YEAR"`
- `"BEEKEEPER" -> "PACKAGE BEES"`
- `"BEEKEEPERS' CLUB" -> "BEEKEEPING"`

Subgraph size:

- `124 entities / 167 relations / 117 sources`

## 2. Legal: strong drift case

Selected group: `q003`  
Base query:

`Which entities have the most complex compliance requirements based on the dataset?`

This group is one of the worst in the legal dataset:

- `high_keyword_jaccard_avg = 0.0533`
- `low_keyword_jaccard_avg = 0.0111`
- `low_keyword_entry_consistency_local_jaccard_avg = 0.2461`
- `high_keyword_entry_consistency_global_jaccard_avg = 0.243`

### 2.1 Base query

Query:

`Which entities have the most complex compliance requirements based on the dataset?`

High-level keywords:

- `Entities`
- `Compliance requirements`
- `Complexity`
- `Dataset`

Low-level keywords:

- `Regulatory frameworks`
- `Industry standards`
- `Legal requirements`
- `Data protection`
- `Compliance audits`

Local entry top-8:

- `"REGULATORY COMPLIANCE"`
- `"SELF-REGULATORY ORGANIZATIONS"`
- `"REGULATORY AUTHORITIES"`
- `"FEDERAL, STATE AND FOREIGN STATUTES"`
- `"REGULATORY REQUIREMENTS"`
- `"REGULATORY OR JURISDICTIONAL PROCEEDINGS"`
- `"HEALTH REGULATORY LAWS"`
- `"REGULATORY DATA"`

Global entry top-8:

- `"GOVERNMENTAL REQUIREMENTS" -> "PROPERTY"`
- `"CODE" -> "LEGAL REQUIREMENTS"`
- `"ENTITY" -> "SANCTIONS"`
- `"GOVERNMENTAL ENTITY" -> "IT SYSTEMS"`
- `"EUROPEAN UNION" -> "PARTNERSHIP ENTITIES"`
- `"GOVERNMENTAL ENTITY" -> "TRANSACTIONS"`
- `"ENTITY" -> "GOVERNMENTAL AUTHORITY"`
- `"SECURITIES LAWS" -> "UNITS"`

Subgraph size:

- `127 entities / 203 relations / 764 sources`

### 2.2 Rewrite 4

Query:

`Which entities have the toughest compliance requirements?`

High-level keywords:

- `Compliance requirements`
- `Regulatory burden`
- `Entity oversight`

Low-level keywords:

- `Corporations`
- `Financial institutions`
- `Healthcare providers`
- `Government agencies`
- `Legal obligations`

Local entry top-8:

- `"OTHER FINANCIAL REPORTING OBLIGATIONS"`
- `"GOVERNMENTAL ENTITY"`
- `"OBLIGATIONS"`
- `"GOVERNMENTAL AUTHORITIES"`
- `"LEGAL OBLIGATIONS"`
- `"FEDERAL, STATE, FOREIGN, REGIONAL, MUNICIPAL REGULATORY BODIES"`
- `"LIABILITY"`
- `"THE COMPANY"`

Global entry top-8:

- `"ENTITY" -> "GOVERNMENTAL AUTHORITY"`
- `"ENTITY" -> "SANCTIONS"`
- `"COMPANY" -> "SEC"`
- `"GOVERNMENTAL REQUIREMENTS" -> "PROPERTY"`
- `"COMPANY" -> "GOVERNMENTAL ENTITIES"`
- `"GOVERNMENTAL ENTITY" -> "THE COMPANY"`
- `"COMPANY" -> "GOVERNMENTAL BODY"`
- `"COMPLIANCE DOCUMENTATION" -> "GOVERNMENTAL AUTHORITY"`

Subgraph size:

- `170 entities / 96 relations / 836 sources`

### 2.3 What this shows

These two questions are also semantically aligned to a human. But the keyword shift is severe:

- base low keywords stay abstract and regulation-centered:
  `regulatory frameworks`, `industry standards`, `compliance audits`
- rewrite low keywords jump to sector/entity prototypes:
  `corporations`, `financial institutions`, `healthcare providers`, `government agencies`

That immediately moves graph entry:

- base local entry is concentrated on general regulatory concepts
- rewrite local entry shifts toward organization/government/company obligation nodes

The global entry also drifts from general legal requirement structure into company- and authority-linked edges such as:

- `"COMPANY" -> "SEC"`
- `"COMPANY" -> "GOVERNMENTAL ENTITIES"`
- `"GOVERNMENTAL ENTITY" -> "THE COMPANY"`

This is a strong visual example that keyword drift is not just a low Jaccard number. It changes where the graph search begins, and therefore changes the evidence region the model will summarize from.

