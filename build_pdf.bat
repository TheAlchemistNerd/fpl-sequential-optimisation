@echo off
setlocal enabledelayedexpansion

echo Building: Predicting FPL Using Sequential Optimisation.pdf...

pandoc --lua-filter=mermaid.lua ^
       -s --pdf-engine=xelatex ^
       -H modern-preamble.tex ^
       --citeproc ^
       --bibliography=docs/thesis/references.bib ^
       -V geometry:margin=1in ^
       -V linkcolor:DodgerBlue ^
       -V title="Optimal Sequential Decision Making in the Fantasy Premier League" ^
       -V subtitle="A Unified Framework for Intertemporal Stochastic Optimal Control via Bayesian Nonparametrics and Deep Reinforcement Learning" ^
       -V author="Nevil Maloba\\ \small GitHub: \href{https://github.com/TheAlchemistNerd}{TheAlchemistNerd} \ \ nevillemaloba@yahoo.com" ^
       --toc ^
       docs/thesis/chapters/chapter_00_abstract.md ^
       docs/thesis/chapters/chapter_01_introduction_problem_formulation.md ^
       docs/thesis/chapters/chapter_02_literature_review.md ^
       docs/thesis/chapters/chapter_03_data_acquisition_feature_engineering.md ^
       docs/thesis/chapters/chapter_04_bayesian_nonparametrics.md ^
       docs/thesis/chapters/chapter_05_hdp_gmm_gaussian_wishart.md ^
       docs/thesis/chapters/chapter_06_hdp_hmm_temporal_dynamics.md ^
       docs/thesis/chapters/chapter_07_posterior_inference_mcmc.md ^
       docs/thesis/chapters/chapter_08_constrained_optimization.md ^
       docs/thesis/chapters/chapter_09_thompson_sampling.md ^
       docs/thesis/chapters/chapter_10_ddqn_bellman_mfg.md ^
       docs/thesis/chapters/chapter_11_mlops_deployment.md ^
       docs/thesis/chapters/chapter_12_experimental_results.md ^
       docs/thesis/chapters/chapter_13_conclusion.md ^
       docs/thesis/chapters/appendices.md ^
       -o "Optimal Sequential Decision Making in the Fantasy Premier League.pdf"

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] PDF has been generated successfully.
) else (
    echo [ERROR] Build failed. Check the output above.
)

pause
