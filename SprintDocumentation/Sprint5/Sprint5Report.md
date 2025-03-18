# Sprint 5 Report (Feb 10th to March 17th)

## [Video](https://drive.google.com/file/d/1zmXLqE_HafRMHIv385-sAVHe6lj9_mZN/view?usp=sharing)

## What's New (User Facing)
 * Installation process revised with updated package dependencies
 * New "programming" prompt type implementation
 * Enhanced error handling and edge case management across evaluation modules

## Work Summary (Developer Facing)
This sprint was dedicated to enhancing our evaluation modules and improving system robustness. We focused on developing comprehensive test cases for the Creativity metrics, ensuring robust end-to-end coverage for all 18 metrics, including a unique approach for the Divergent Association Task (DAT) metric. The team undertook significant refactoring of the testing infrastructure to decouple tests and implement independent setup and teardown functions, enhancing test reliability and reproducibility. Concurrently, we expanded the Ethical Consideration Module with additional evaluation criteria and support for parallel processing, which significantly reduced evaluation times. Architecture improvements were made by refactoring evaluators into abstract classes, allowing better support for multiprocessing and solving lingering issues from previous sprints. The installation process was streamlined to reflect new dependencies, and the new HackerEarth question prompt type was integrated, facilitating more versatile prompt management.

## Unfinished Work
 * Additional test coverage is needed for edge cases in the Ethical Consideration Module due to the complexity of evaluating diverse ethical scenarios.
 * Further refinement is required for performance optimization in parallel processing during large-scale evaluations.
 * Link the evaluation modules backend to the LLMEval cli interface "frontend"

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:

 * [Fix installation process](https://github.com/RyderSwanson/LLMEval/pull/92)
 * [Enhanced error handling](https://placeholder.github.com)
 
## Incomplete Issues/User Stories
Here are links to issues we worked on but did not complete in this sprint:
 
 * [Writing a Research Paper](https://github.com/users/RyderSwanson/projects/2/views/1?pane=issue&itemId=97340991&issue=RyderSwanson%7CLLMEval%7C84)
 We were given new metrics to add, delaying the research paper.
 * [Add Programming metric](https://github.com/RyderSwanson/LLMEval/tree/code-evaluation-metric)
 This issue was still in progress when the sprint ended.

## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
 * [EvaluationModules.py](https://github.com/RyderSwanson/LLMEval/blob/main/EvaluationModules.py)
 * [LLMEval.py](https://github.com/RyderSwanson/LLMEval/blob/main/LLMEval.py)
 * [pyproject.toml](https://github.com/RyderSwanson/LLMEval/blob/main/pyproject.toml)
 
## Retrospective Summary
Here's what went well:
 * Successfully implemented comprehensive test coverage for Creativity metrics.
 * Improved system architecture through better abstraction and multiprocessing support.
 * Added new prompt type.
 * Enhanced installation process reliability.
 
Here's what we'd like to improve:
 * Expand test coverage for edge cases, particularly in ethical evaluations.
 * Optimize performance of parallel processing for large-scale evaluations.
 * Improve integration testing procedures.
 * Update documentation to reflect new features and enhancements.
  
Here are changes we plan to implement in the next sprint:
 * Link the evaluation modules backend to the LLMEval CLI interface frontend.
 * Set up automated integration testing pipeline.
 * Continue optimization efforts for parallel processing.
 * Achieve full edge case coverage in ethical evaluation testing.
 * Enhance documentation with detailed feature descriptions and user guidance.
