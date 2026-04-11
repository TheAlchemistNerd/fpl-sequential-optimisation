@echo off
setlocal enabledelayedexpansion

echo Building: Predicting FPL Using Sequential Optimisation.pdf...

pandoc --lua-filter=mermaid.lua ^
       -s --pdf-engine=xelatex ^
       -H modern-preamble.tex ^
       --citeproc ^
       --bibliography=docs/references.bib ^
       -V geometry:margin=1in ^
       -V linkcolor:DodgerBlue ^
       -V author="Nevil Maloba\\ \small GitHub: TheAlchemistNed \ \ nevillemaloba@gmail.com" ^
       --toc ^
       docs/chapter_00_abstract.md ^
       docs/chapter_01_introduction_problem_formulation.md ^
       docs/chapter_02_literature_review.md ^
       docs/chapter_03_data_acquisition_feature_engineering.md ^
       docs/chapter_04_bayesian_nonparametrics.md ^
       docs/chapter_05_hdp_gmm_gaussian_wishart.md ^
       docs/chapter_06_hdp_hmm_temporal_dynamics.md ^
       docs/chapter_07_posterior_inference_mcmc.md ^
       docs/chapter_08_thompson_sampling.md ^
       docs/chapter_09_ddqn_bellman_mfg.md ^
       docs/chapter_10_constrained_optimization.md ^
       docs/chapter_11_mlops_deployment.md ^
       docs/chapter_12_experimental_results.md ^
       docs/chapter_13_conclusion.md ^
       docs/appendices.md ^
       -o "Predicting FPL Using Sequential Optimisation.pdf"

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] PDF has been generated successfully.
) else (
    echo [ERROR] Build failed. Check the output above.
)

pause
