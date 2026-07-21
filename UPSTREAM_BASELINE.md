# Upstream baseline and comparison boundary

- Upstream repository: `https://github.com/LMS4681/CNN-RL.git`
- Approved baseline SHA: `cd4e14fc1725a4ff159e59d6874d3602f3b65a06`
- Fixed scenarios SHA-256: `913cac9046dec8164ef65da60275522f7127de5ea775b1c5a6b6aac255716271`
- Split manifest SHA-256: `601bd6143ed8890577e5ff34921241d36fd6a0e99c4bdab4e26152ab168178f8`
- Public comparison immutable tag: `overnight-v1`
- Dependency lock SHA256: `37634576e34043d169cf24bfc0cc2261818dc65b9358d4b9b2e46ab614d0bdda`
- Inherited `AllocRL/data` tree OID: `0140dfe704c607045da2f20faa32a0141e7bcc9b`

The tracked `AllocRL/data` tree is inherited unchanged from the same owner's
public baseline at the approved commit; this comparison adds or modifies no
files under that directory.

The comparison repository is a separate publication boundary. Comparison-only
commits must never be placed onto the original upstream main/history. The
notebook verifies the immutable tag, fixed inputs, and this lock hash before it
runs the experiment.
