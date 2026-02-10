# CSC 475 Project. DP1: Lyrics-Based Autotagging

## Project Goal
This project investigates automatic music tag prediction from song lyrics using multi-label classification. The main goal is to build and evaluate a reproducible end-to-end pipeline that maps lyrics text to one or more semantic tags (e.g., genre, mood, theme, feeling, etc.). The project will use the Music4All dataset (local copy) and focuses on text-first modeling with clear baselines, error analysis, and practical evaluation metrics for imbalanced multi-label learning.

This document is in progress and will evolve continuously throughout the development process.

## Project Proposal
### Problem Statement
Music has more than one quality about it that can be assigned as a tag. This project targets lyrics-based auto-tagging and asks:
1. How well can text-only features predict multiple music tags?
2. Which model families provide the best tradeoff between performance, interpretability, and training/inference cost?
3. What are the main failure modes (label imbalance, ambiguous tags, sparse lyrics) and how can they be mitigated?
4. Can we get infer all the genres from the lyrics, or some can only be infered by the sound?

### Scope
In scope:
1. Data loading and preprocessing for Music4All lyrics/metadata subset.
2. Multi-label experiments with classical ML baselines and at least one neural model.
3. Evaluation using micro/macro F1, Hamming loss, subset accuracy, and per-tag analysis.
4. Error analysis and ablation studies on preprocessing and representation choices.

Scope can be adjusted.

## Team Structure
### Lev Chshemelinin
Solo project.

Roles handled by me:
1. Project management and timeline tracking.
2. Dataset preparation and quality checks.
3. Model implementation, experimentation, and evaluation.
4. Literature review and writing.

## Timeline and Deadlines (until end of semester)
Target submission window: end of March 2026.

### Week 1 (February 9-February 15, 2026): Setup and Data Audit
1. Confirm local dataset schema, file integrity, and split strategy.
2. Build reproducible data-loading and preprocessing scripts.
3. Define initial tag set and filtering thresholds (minimum sample count per tag).

Deliverables:
1. Data dictionary draft.
2. Clean preprocessing pipeline with saved intermediate artifacts.

### Week 2 (February 16-February 22, 2026): Baseline Models
1. Implement TF-IDF + linear models for multi-label classification (One-vs-Rest Logistic Regression, Linear SVM).
2. Establish baseline metrics with stratified or iterative split protocol.
3. Create first confusion-style/tag-level error visualizations.

Deliverables:
1. Baseline results table.
2. Initial observations on imbalance and weak tags.

### Week 3 (February 23-March 1, 2026): Feature and Pipeline Expansion
1. Compare vectorizers (word vs. character n-grams).
2. Evaluate text normalization options (lemmatization, stop-word policy, min_df/max_df).
3. Add calibration/threshold tuning for better multi-label decision quality.

Deliverables:
1. Ablation results for feature engineering.
2. Updated experiment tracking sheet.

### Week 4 (March 2-March 8, 2026): Neural Baseline
1. Implement at least one neural text model (e.g., embedding + BiLSTM/CNN).
2. Compare neural and classical methods under same split and metrics.
3. Analyze runtime and compute cost.

Deliverables:
1. Neural baseline performance report.
2. Resource and timing comparison table.

### Week 5 (March 9-March 15, 2026): Error Analysis and Robustness
1. Inspect false positives/negatives by tag and lyric length bins.
2. Test class weighting, resampling, and threshold strategies.
3. Identify reliable/fragile tags and explain likely causes.

Deliverables:
1. Error analysis section draft.
2. Revised model recommendations.

### Week 6 (March 16-March 22, 2026): Final Model Selection
1. Lock best-performing pipeline using validation results.
2. Run final test evaluation and generate final plots/tables.
3. Perform reproducibility checks (seed, environment, rerun consistency).

Deliverables:
1. Final experiment package.
2. Reproducibility checklist.


### Week 7 (March 23-March 31, 2026): Report and Submission
1. Finalize README progress content for course deliverables.
2. Prepare structure that can transfer into ISMIR paper format (LaTeX).
3. Verify code, figures, references, and bibliography consistency.

Deliverables:
1. Final course submission package.
2. Report-ready text sections for migration to LaTeX.


## Tools, Libraries, and Infrastructure
### Core Software
1. Python 3.x
2. scikit-learn (multi-label wrappers, vectorizers, metrics)
3. pandas / numpy (data processing)
4. matplotlib / seaborn (plots and diagnostics)
5. PyTorch or TensorFlow/Keras (neural baseline)
6. Jupyter notebooks + scripts for experiments
7. Git/GitHub for version control and progress tracking


### Referenced Technical Guides
1. scikit-learn multi-label classification documentation.
2. scikit-learn text processing tutorial.


## Datasets
### Primary Dataset
1. Music4All (local copy)
2. Source: https://sites.google.com/view/contact4music4all


### Planned Data Handling
1. Use lyrics and selected tag/metadata fields.
2. Standardize text preprocessing (tokenization, case normalization, optional lemmatization).
3. Filter extremely rare tags to control sparsity.
4. Preserve held-out evaluation protocol for unbiased final results.


## Methodology Overview
1. Formulate as supervised multi-label text classification.
2. Start from interpretable baselines:
   - TF-IDF + One-vs-Rest Logistic Regression.
   - TF-IDF + Linear SVM (or SGDClassifier hinge/log loss).
3. Expand to neural baseline:
   - Sequence model over lyrics (e.g., CNN or BiLSTM with sigmoid output layer).
4. Optimize classification thresholds for multi-label decisions.
5. Report both aggregate and per-label metrics; include runtime statistics.


## Evaluation Plan
Primary metrics:
1. Micro-F1
2. Macro-F1
3. Hamming loss
4. Subset accuracy
5. Precision@k / Recall@k (if applicable to final output format)

Secondary analysis:
1. Per-tag precision/recall/F1.
2. Impact of label frequency imbalance.
3. Error slices by lyric length and vocabulary coverage.

## Risks and Mitigations
1. Severe class imbalance:
   - Mitigation: class weighting, threshold tuning, rare-tag filtering.
2. Noisy/incomplete lyrics:
   - Mitigation: data cleaning and explicit missing-data policy.
3. Overfitting with sparse high-dimensional features:
   - Mitigation: regularization, validation-based model selection, ablations.
4. Limited compute/time:
   - Mitigation: prioritize strongest baselines first, then add one neural model.

## Related Work
Lyrics-based and multimodal MIR tagging has progressed from feature-engineered classifiers to deep architectures. Early work demonstrated semantic annotation and retrieval using supervised learning over music-related features and tag spaces (Turnbull et al., 2008). Subsequent studies explored combining lyrics and audio representations for genre and tag prediction, showing that lyric information can be complementary to acoustic cues (Mayer and Rauber, 2011). Larger curated datasets such as Music4All made broader benchmarking possible and supported modern supervised pipelines (Santana et al., 2020).

On the modeling side, classical text pipelines using bag-of-words/TF-IDF with linear classifiers remain strong baselines, especially with sparse high-dimensional text and limited training resources (Joachims, 1998; Nigam et al., 2000). For multi-label learning specifically, problem-transformation methods such as Binary Relevance and Classifier Chains provide practical baselines, while ranking-based and deep models can better exploit label dependencies under sufficient data (Tsoumakas and Katakis, 2007; Read et al., 2011; Zhang and Zhou, 2014).

Deep learning in MIR and text classification introduced CNN/RNN/CRNN approaches that capture richer structure than purely linear pipelines (Kim, 2014; Choi et al., 2017). More recent developments in contextual language modeling and transfer learning suggest potential gains for lyric understanding and semantic tagging, though computational cost and interpretability tradeoffs remain central (Devlin et al., 2019). This project positions itself as a careful, reproducible comparison across strong classical baselines and one neural lyric model, with emphasis on robust evaluation and error analysis for multi-label music tagging.

## Bibliography, Relevant Works
1. Turnbull, D., Barrington, L., Torres, D., and Lanckriet, G. (2008). Semantic annotation and retrieval of music and sound effects. IEEE Transactions on Audio, Speech, and Language Processing, 16(2), 467-476.
2. Mayer, R., and Rauber, A. (2011). Musical genre classification by ensembles of audio and lyrics features. In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR).
3. Santana, I. A. P., Pinhelli, F., Donini, J., Catharin, L., Mangolin, R. B., and Feltrim, V. D. (2020). Music4All: A new music database and its applications. In 2020 International Conference on Systems, Signals and Image Processing (IWSSIP).
4. Choi, K., Fazekas, G., Sandler, M., and Cho, K. (2017). Convolutional recurrent neural networks for music classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
5. Tsoumakas, G., and Katakis, I. (2007). Multi-label classification: An overview. International Journal of Data Warehousing and Mining, 3(3), 1-13.
6. Zhang, M.-L., and Zhou, Z.-H. (2014). A review on multi-label learning algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8), 1819-1837.
7. Read, J., Pfahringer, B., Holmes, G., and Frank, E. (2011). Classifier chains for multi-label classification. Machine Learning, 85, 333-359.
8. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In ECML.
9. Nigam, K., McCallum, A., Thrun, S., and Mitchell, T. (2000). Text classification from labeled and unlabeled documents using EM. Machine Learning, 39, 103-134.
10. Sebastiani, F. (2002). Machine learning in automated text categorization. ACM Computing Surveys, 34(1), 1-47.
11. Kim, Y. (2014). Convolutional neural networks for sentence classification. In EMNLP.
12. Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
13. Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global vectors for word representation. In EMNLP.
14. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.
15. Schedl, M., Gómez, E., and Urbano, J. (2014). Music information retrieval: Recent developments and applications. Foundations and Trends in Information Retrieval, 8(2-3), 127-261.
16. Oramas, S., Barbieri, F., Nieto, O., and Serra, X. (2017). Multimodal deep learning for music genre classification. Transactions of the International Society for Music Information Retrieval, 1(1), 4-21.
17. Hu, X., Downie, J. S., West, K., and Ehmann, A. F. (2008). Exploring mood metadata: Relationships with genre, artist and usage metadata. In ISMIR.
18. Laurier, C., Grivolla, J., and Herrera, P. (2008). Multimodal music mood classification using audio and lyrics. In 7th International Conference on Machine Learning and Applications (ICMLA).


## Personal Objectives
## Lev Chshemelinin
Objective 1 (Build a functional end-to-end lyrics-based multi-label tagging pipeline on Music4All)
1. PI1 (basic): load lyrics and tag metadata from the local Music4All copy and produce a clean modeling table.
2. PI2 (basic): implement a reproducible preprocessing pipeline and fixed train/validation/test split.
3. PI3 (expected): train at least two baseline multi-label classifiers and report micro/macro F1.
4. PI4 (expected): generate per-tag evaluation summaries and identify at least five high-error tags.
5. PI5 (advanced): implement a threshold-tuning strategy that improves macro-F1 over untuned baselines.

Objective 2 (Perform comparative experiments and document engineering tradeoffs)
1. PI1 (basic): benchmark TF-IDF word n-gram and character n-gram features with one linear classifier.
2. PI2 (basic): track training and inference times for each experimental run.
3. PI3 (expected): compare at least three model configurations and present results in a single summary table.
4. PI4 (expected): conduct an ablation on preprocessing choices (e.g., stop-word policy, min_df, lemmatization).
5. PI5 (advanced): add one neural model and analyze performance vs. compute tradeoff relative to best linear model.

Objective 3 (Produce a report-ready research narrative with reproducible evidence)
1. PI1 (basic): maintain an experiment log with parameters, seeds, and metric outputs.
2. PI2 (basic): draft related work text connecting at least 15 references to project design decisions.
3. PI3 (expected): include error analysis with concrete examples of false positives and false negatives.
4. PI4 (expected): provide at least three publication-quality figures/tables for final write-up.
5. PI5 (advanced): package scripts/configuration so final results can be rerun with minimal manual changes.

