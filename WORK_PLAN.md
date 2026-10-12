# Project Review and Reorganization Work Plan

Status: review and reorganization completed on 2026-10-12. The full run passed 458 tests with the Torch-dependent DDQN module excluded; after the final MILP consolidation and EDA/API cleanup, 32 focused selection tests passed and the directly affected EDA/API checks were rerun.

Work through these steps in order. Do not decide on a reorganization or edit project content before completing the review steps.

1. Read every Markdown file in `docs/` line by line, including the chapters and appendices, and record what each says the project is meant to do.
2. Read every source file under `src/` line by line, follow the actual data flow, and note how modules call one another. Compare what the code does with what the docs claim.
3. Build an evidence-based picture of the project's objective and current behavior from those files. Do not settle on a reorganization plan or make edits before that review is complete.
4. Inspect and run the existing tests to help establish how the current code behaves. Leave test changes for the reorganization stage.
5. Only after finishing the `docs/` and `src/` review, read every root-level Markdown file line by line. Then assess which documents are useful, outdated, overlapping, or redundant.
6. Use the completed review to decide what should be reorganized across `docs/` and `src/`, keeping intended research ideas distinct from features the code actually implements.
7. Reorganize and simplify `docs/` and `src/`: remove duplicated, stale, unused, or bloated material when the completed review supports removal; consolidate repeated explanations and overlapping implementations; update affected references; preserve distinct research ideas and useful behavior rather than retaining litter by default. Review redundancy within the thesis chapters too, and order the research concepts progressively from foundations to more difficult methods.
8. Address tests at that stage: update or add them as needed for the reorganized code, then run the relevant checks and report what passed or remains unresolved.

## Ordering note

Root-level Markdown was read too early in an earlier pass. Those notes were set aside, and all root-level Markdown files were reread for the redundancy assessment after the `docs/` and `src/` review.

## Cleanup emphasis

The reorganization should materially reduce documentation, research, and code bloat. Treat redundant documents or chapter sections, stale claims, dead code, compatibility clutter, and overlapping implementations as candidates for removal or consolidation, using the completed line-by-line review and test evidence to guide each decision. Sequence the retained research so prerequisites are introduced before advanced models and decision methods.
