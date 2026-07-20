# Upstream baseline and comparison boundary

- Upstream repository: `https://github.com/LMS4681/CNN-RL.git`
- Approved baseline SHA: `cd4e14fc1725a4ff159e59d6874d3602f3b65a06`
- Fixed scenarios SHA-256: `6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814`
- Split manifest SHA-256: `d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df`
- Public comparison immutable tag: `overnight-v1`
- Dependency lock SHA256: `2098a1d132dde6e3255b0e7be6193edb3b09f758565aa319837afd53dbdf4bd7`

The comparison repository is a separate publication boundary. Comparison-only
commits must never be placed onto the original upstream main/history. The
notebook verifies the immutable tag, fixed inputs, and this lock hash before it
runs the experiment.
