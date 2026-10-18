# Building Intelligence: An Adventure in AI and Data

*Lau Wen Jun*

![](https://user-images.githubusercontent.com/74038190/240906093-9be4d344-6782-461a-b5a6-32a07bf7b34e.gif)

## About Me 👦

👋 Hi, I'm Wen Jun — an AI Engineer working on LLM-powered knowledge systems and enterprise-scale data pipelines at Lenovo, with a background spanning machine learning, statistics, and data engineering. I enjoy building end-to-end systems: from raw data through models to working applications people can actually use.

Current focus: LLM/RAG systems, multi-agent orchestration, computer vision, and reinforcement learning applied to real decision problems.

## Live Demos :rocket:

Click and try them in your browser — no installation needed.

### Flagship

- **[Ask WJ — Source-grounded Personal AI Assistant](https://wenjun78.github.io/portfolio/ask-wj.html)** ([source](https://github.com/wenjun78/ask_wj)) (FastAPI, hybrid BM25 + embedding retrieval, RRF, PostgreSQL, SSE) — an end-to-end, auditable RAG application with cited answers, an owner dashboard, prompt versioning, privacy and rate-limit controls, retrieval evaluation fixtures and automated CI checks. Case study documents current limitations; no unverified user metrics or live deployment claims.
- **[MOLOA — Multi-Agent Loan Origination & Lifecycle Decisioning](https://wenjun78.github.io/MOLOA/demo/moloa_demo.html)** (PyTorch, XGBoost) — Master's thesis: a multi-agent AI system scoring 50,000 real Freddie Mac mortgages across origination, monitoring, and distress intervention — learned attention routing, causal treatment-effect modelling, and honestly reported negative results.
- **[RefVision — Referee Signal Recognition](https://wenjun78.github.io/FIFA/RefVision/docs/refvision.html)** (MediaPipe, JavaScript) — a camera reads the referee's signals (free kicks, penalty, advantage, VAR, cards) and logs each decision in real time — fully on-device, no cloud, no LLM.
- **[Penalty Save Simulation](https://wenjun78.github.io/FIFA/PenaltySaver/vr_penalty_saver.html)** (Three.js, WebXR) — first-person VR goalkeeper simulation with physics-based ball flight and research-calibrated penalty-anticipation cues.

## Selected AI & Machine Learning Projects

A collection of applied AI and machine learning projects covering computer vision, deep learning, multimodal AI, model evaluation and explainable AI. Each project has a corresponding Hugging Face live demo.

### 1. [Medical Visual Question Answering with BiomedCLIP](https://huggingface.co/spaces/wenjun78/medvqa)

**Multimodal AI | Vision-Language Models | Academic Team Project**

**[🤗 Try the Hugging Face Demo](https://huggingface.co/spaces/wenjun78/medvqa)**

Developed and evaluated a medical visual question-answering approach combining medical images and natural-language questions to predict clinically relevant answers.

- Compared a *ResNet50–Transformer* baseline against a biomedical vision-language model based on *BiomedCLIP*.
- Evaluated performance using the *SLAKE* dataset, including **1,061 English-language test question-answer pairs**.
- Achieved **83.88% exact-match accuracy** with the BiomedCLIP-based model, outperforming the baseline's **80.87%**.
- Investigated model ensembles, semantic evaluation and errors involving medical terminology.

**Technologies:** Python, PyTorch, BiomedCLIP, Transformers, NLP, Computer Vision

### 2. [Malaysian Sign Language Recognition Using MediaPipe and BiLSTM](https://huggingface.co/spaces/wenjun78/gesture)

**Deep Learning | Computer Vision | Sequence Modelling | Academic Team Project**

**[🤗 Try the Hugging Face Demo](https://huggingface.co/spaces/wenjun78/gesture)**

[![Malaysian Sign Language Recognition — live webcam demo](assets/malaysian_sign_language_portfolio.png)](https://huggingface.co/spaces/wenjun78/gesture)

Developed a sign-language recognition model that converts sequences of human body, hand and facial landmarks into gesture predictions.

- Extracted landmark sequences using *MediaPipe Holistic* and trained a *Bidirectional LSTM* network to recognise **30 sign-language gestures**.
- Processed **2,975 gesture sequences**, using training, validation and test splits.
- Achieved **97.65% test accuracy** and implemented video-based gesture prediction.
- Evaluated predictions using classification metrics and confusion matrices.

**Technologies:** Python, PyTorch, MediaPipe, BiLSTM, OpenCV, Scikit-learn

### 3. [Pneumonia Detection with Explainable Deep Learning](https://huggingface.co/spaces/wenjun78/pneumonia_detection)

**Medical Imaging | CNNs | Explainable AI | Academic Project**

**[🤗 Try the Hugging Face Demo](https://huggingface.co/spaces/wenjun78/pneumonia_detection)**

Developed and compared deep learning models to classify chest X-ray images as normal or pneumonia, with explainable AI techniques to visualise model predictions.

- Benchmarked **5 deep learning architectures**: *VGG16, ResNet50, DenseNet121, EfficientNetB0* and *ConvNeXt-Tiny*.
- Achieved **94.23% test accuracy** and **95.42% pneumonia-class F1** with *VGG16* on **624 test images**.
- Applied *Grad-CAM++* and *LIME* to visualise image regions associated with predictions.
- Compared precision, recall, F1, ROC-AUC and PR-AUC to assess classification performance.

**Technologies:** Python, TensorFlow, Keras, Transfer Learning, Grad-CAM++, LIME

### 4. [Blood Cell Classification with Explainable AI](https://huggingface.co/spaces/wenjun78/blood_cell)

**Image Classification | Deep Learning | XAI | Academic Project**

**[🤗 Try the Hugging Face Demo](https://huggingface.co/spaces/wenjun78/blood_cell)**

Developed a deep learning comparison study for classifying microscopic blood-cell images into eight categories, with a focus on model performance and interpretability.

- Compared *VGG16, DenseNet121, EfficientNetB3* and *ConvNeXt-Tiny* across **8 blood-cell classes**.
- Achieved **96.78% test accuracy** and **96.65% macro F1** with *VGG16* on **2,140 test images**.
- Applied *Grad-CAM* heatmaps and counterfactual explanations to examine model predictions.
- Evaluated classification performance using precision, recall, F1 and ROC/PR curves.

**Technologies:** Python, TensorFlow, Keras, CNNs, Grad-CAM, Counterfactual Explainability

### 5. [Bone Fracture Detection and Classification Using YOLO12](https://huggingface.co/spaces/wenjun78/bonefracture)

**Object Detection | Medical Imaging | Academic Team Project**

**[🤗 Try the Hugging Face Demo](https://huggingface.co/spaces/wenjun78/bonefracture)**

Developed a YOLO-based object detection approach to locate and classify bone-fracture patterns in medical images.

- Trained *YOLO12n* on the **1,539-image HBFMID dataset**, covering **10 classes**, including healthy and fracture categories.
- Achieved **94.09% mAP@50**, **50.02% mAP@50–95** and **96.12% precision** on the **64-image test set**.
- Evaluated per-class performance, detection confidence and predicted bounding boxes.
- Visualised fracture detections to assess localisation and classification quality.

**Technologies:** Python, Ultralytics YOLO12, PyTorch, OpenCV, Object Detection

---

## Publications :page_facing_up:

[Explore the publications page](https://wenjun78.github.io/portfolio/publications.html) for the peer-reviewed article, research summary, DOI and citation download.

Lau, W., Abdul Rahman, H. (2026). *Predicting academic performance through machine learning: integrating demographic, psychological, and behavioral predictors using explainable AI.* Asia Pacific Education Review. [doi:10.1007/s12564-026-10124-y](https://doi.org/10.1007/s12564-026-10124-y)

## Dashboards & Analytics :bar_chart:

[Explore professional dashboard and analytics case studies](https://wenjun78.github.io/portfolio/dashboards.html).

- **Education Statistics 2022–2024** — Ministry of Education, Brunei
- **Employer Satisfaction Survey 2022–2023** — Ministry of Education, Brunei
- **Graduate Employment Study 2022–2024** — Ministry of Education, Brunei
- **User Monetisation & Marketing ROI Dashboard** — BPO

These summaries describe internal work; employer dashboards and source datasets are not publicly shared.

## More Projects :writing_hand:

Additional work in reinforcement learning, predictive modelling, experimentation, explainable AI and applied analytics. Linked entries open the corresponding analysis or interactive dashboard.

### Clinical Decision Support (Research)

#### ACS Triage Reinforcement Learning Agent

**Reinforcement Learning | Clinical Risk Modelling | Research Project**

Explored reinforcement learning for recommending patient disposition following acute coronary syndrome (ACS) triage using structured clinical information.

- Modelled **discharge, observation and admission** as possible decisions using clinical features such as vital signs, ECG findings and troponin results.
- Applied *Deep Q-Network (DQN)* methods with safety-weighted rewards that penalise missed major adverse cardiac events (MACE).
- Examined trade-offs between avoiding missed high-risk cases and unnecessary admissions. **Academic research only, not a clinical decision tool.**

**Technologies:** Python, PyTorch, Reinforcement Learning, DQN, Clinical Data Analysis

### Explainable AI

#### [NASA Turbofan Engine Remaining Useful Life Prediction](https://wenjun78.github.io/portfolio/XAI/xai_dashboard.html)

**Predictive Maintenance | Time-Series Forecasting | Explainable AI**

**[📊 View Interactive XAI Dashboard](https://wenjun78.github.io/portfolio/XAI/xai_dashboard.html)**

Built an interactive dashboard that explains remaining useful life (RUL) predictions and helps explore engine maintenance risk.

- Explored the *NASA C-MAPSS FD001* dataset with **100 test engines**, using an *LSTM-attention* predictive-maintenance approach.
- Visualised predicted versus actual RUL, engine risk categories and sensor-degradation trends.
- Added feature-importance views, what-if analysis and per-engine prediction details to help interpret model outputs.

**Technologies:** Python, LSTM, Attention, Explainable AI, Time-Series Analysis, Interactive Dashboards

### Data Science & Analytics

#### [Loan Repayment Prediction](https://wenjun78.github.io/portfolio/project/LoanRepay/DS_LoanRepayment_Lau.html)

**Credit Risk | Supervised Classification | Model Comparison**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/LoanRepay/DS_LoanRepayment_Lau.html)**

Analysed loan application, payment and underwriting data to develop predictive models for loan-quality classification.

- Explored **577,682 loan applications**, including a reported **459,393 unique applicants**, and joined application, payment and underwriting records.
- Compared *Logistic Regression, Decision Tree, Random Forest* and *XGBoost*, including feature engineering, feature selection and a stratified train/test split.
- The notebook reports **98.42% accuracy, 84.63% F1 and 91.82% recall** for *XGBoost* on its evaluation set.
- **Caution:** Some selected predictors, such as funding or payment-status fields, may become available only after a lending decision, creating potential target leakage. The scores should not be interpreted as validated pre-origination credit-risk performance.

**Technologies:** Python, Pandas, Scikit-learn, XGBoost, Feature Engineering, Classification

#### [Car Price Prediction](https://wenjun78.github.io/portfolio/project/Carsale/prediction_model.html)

**Regression | Feature Engineering | Model Comparison**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/Carsale/prediction_model.html)**

Developed and compared regression models for estimating used-car prices from vehicle characteristics and historical sales data.

- Cleaned and prepared approximately **95,424 vehicle records**, engineering age, mileage and condition-related features.
- Compared *Random Forest, XGBoost* and *Gradient Boosting* using RMSE, MAE and R².
- The reported *Random Forest* evaluation achieved **R² ≈ 0.98** and **MAE ≈ 2,427** in the dataset's price units.

**Technologies:** Python, Scikit-learn, XGBoost, Pandas, Feature Engineering

#### [Oil Palm Production Prediction](https://wenjun78.github.io/portfolio/project/oilpalm/oilproduction_prediction_LauWJ.html)

**Time-Series Forecasting | Agriculture | Predictive Analytics**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/oilpalm/oilproduction_prediction_LauWJ.html)**

Forecasted monthly oil palm production using historical production, weather and plantation data.

- Engineered seasonal, rainfall, fertiliser and production-lag features from **132 monthly observations**.
- Compared regression and tree-based models, including *Random Forest, Gradient Boosting* and *XGBoost*.
- Produced forecasts for January–March 2024; the selected tuned *XGBoost* model reported approximately **14.1% MAPE** on the evaluation split.

**Technologies:** Python, XGBoost, Scikit-learn, Time-Series Feature Engineering, Regression

#### [A/B Testing: UX Redesign on Conversion & Engagement](https://wenjun78.github.io/portfolio/project/abtest_ux/abtest_ux.html)

**Experimentation | Hypothesis Testing | Product Analytics**

**[📊 View A/B Test Report](https://wenjun78.github.io/portfolio/project/abtest_ux/abtest_ux.html)**

Analysed a **synthetic A/B-test dataset** comparing an existing product-page design with a redesigned version.

- Evaluated conversion rate, bounce rate and time on page using proportion tests and a two-sample t-test.
- Found an approximately **22% lower bounce rate** and **18% longer time on page** in the test group.
- Reported that the **conversion difference was not statistically significant**, separating engagement improvements from conversion claims.

**Technologies:** Python, Pandas, SciPy, Statsmodels, Hypothesis Testing

#### [A/B Testing: Discount Impact on GMV & ROI](https://wenjun78.github.io/portfolio/project/abtest_gmv/abtest_gmv.html)

**Experimentation | Commercial Analytics | Statistical Testing**

**[📊 View A/B Test Report](https://wenjun78.github.io/portfolio/project/abtest_gmv/abtest_gmv.html)**

Analysed a **synthetic experiment** comparing a checkout discount with a no-discount control group.

- Evaluated the effect of a **10% discount** on average gross merchandise value (GMV) using a two-sample t-test.
- Recorded **16.32% higher average GMV** in the test group in the simulated dataset.
- Compared the simulated uplift with promotional discount costs to discuss commercial trade-offs; GMV is not the same as net profit.

**Technologies:** Python, Pandas, SciPy, A/B Testing, GMV Analysis

#### [User Segmentation & In-App Purchase Behaviour Analysis](https://wenjun78.github.io/portfolio/project/cusseg/customer_segmentation_20250109.html)

**Customer Analytics | K-Means | Unsupervised Learning**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/cusseg/customer_segmentation_20250109.html)**

Segmented **2,000 synthetic customer records** using demographic, spending and purchase-frequency attributes to explore differentiated marketing strategies.

- Standardised **5 numerical features** and used *K-Means* with elbow and silhouette analysis to assess candidate cluster counts.
- Selected **4 customer segments** for the final interpretation, based on an elbow at k=4, although the **highest silhouette score was approximately 0.48 at k=5**.
- Profiled segment characteristics and visualised cluster separation with *PCA* to propose tailored engagement strategies.

**Technologies:** Python, Pandas, Scikit-learn, K-Means, PCA, Customer Analytics

#### User Journey & Conversion Funnel Analysis

**Product Analytics | Funnel Analysis | Sequential Pattern Mining**

Analysed customer activity sequences to understand how users progressed through different stages of a journey, where drop-offs occurred, and which behaviours were associated with engagement and conversion.

- Mapped stage-to-stage customer transitions using *funnel analysis* and *Sankey diagrams* to visualise common paths and identify drop-off points.
- Explored behavioural relationships with *Apriori association rules* and *sequential-pattern mining*, focusing on frequently occurring actions and event sequences.
- Interpreted patterns across customer groups to inform hypotheses for targeted engagement, personalised promotions and conversion improvements.

**Technologies:** Python, Pandas, Funnel Analysis, Sankey Diagrams, Apriori, Sequential-Pattern Mining

#### [Anomaly Detection in User Transaction Behaviour](https://wenjun78.github.io/portfolio/project/anomaly/Anomaly%20Detection.html)

**Anomaly Detection | Time Series | Statistical & Deep Learning Methods**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/anomaly/Anomaly%20Detection.html)**

Compared statistical, machine learning and deep learning approaches for detecting unusual sales movements in a **366-day synthetic time series containing 57 labelled anomalies**.

- Benchmarked **15+ detection approaches**, including *Isolation Forest, IQR, SVM* and *LSTM Autoencoder*, using precision, recall and F1 against synthetic anomaly labels.
- The notebook reports **0.98 F1, 1.00 precision and 0.96 recall** for *LSTM Autoencoder*; these are experiment-specific results, not evidence of real-world production performance.
- Compared ensemble agreement and false-positive/false-negative trade-offs for anomaly screening.

**Technologies:** Python, Scikit-learn, LSTM Autoencoder, Isolation Forest, Time-Series Analysis

#### [Transaction Security: A/B Testing & Risk Pattern Analysis](https://wenjun78.github.io/portfolio/project/trans_security/transactionsec.html)

**Fraud Analytics | Hypothesis Testing | Classification Evaluation**

**[📊 View Project Analysis](https://wenjun78.github.io/portfolio/project/trans_security/transactionsec.html)**

Evaluated fraud indicators and standard versus extra verification using **1,000 synthetic transaction records**.

- Compared simulated fraud rates with a two-proportion z-test, reporting **78.88%** under standard verification versus **24.38%** under extra verification (**54.5 percentage-point difference**).
- Analysed flagged-transaction performance at **52.00% recall** and **51.41% precision**, highlighting missed fraud and false-positive trade-offs.
- Examined risk-pattern categories to discuss how verification rules can impact legitimate customers.
- **Caution:** The dataset was generated for demonstration, so group differences are not proof that extra verification causes lower fraud rates in a real deployment.

**Technologies:** Python, Pandas, Hypothesis Testing, Fraud Analytics, Confusion Matrix

<!----- **[Web Scraping](https://wenjun78.github.io/portfolio/project/Python/Web_Scraping/Web_Scraping.html)**

### ML Foundations (R notebooks)

[Feature Selection](https://wenjun78.github.io/portfolio/Feature%20Selection.html) · [Regularized Linear Modeling](https://wenjun78.github.io/portfolio/Regularized%20Linear%20Regression.html) · [Dimensionality Reduction](https://wenjun78.github.io/portfolio/Dimensionality%20Reduction.html) · [Lazy Learning](https://wenjun78.github.io/portfolio/Lazy%20Learning.html) · [Naive Bayes](https://wenjun78.github.io/portfolio/Naive%20Bayes.html) · [Decision Tree](https://wenjun78.github.io/portfolio/Decision%20Tree.html) · [Forecasting](https://wenjun78.github.io/portfolio/Forecasting.html) · [Black Box](https://wenjun78.github.io/portfolio/Black%20Box.html) · [Apriori Association Rules](https://wenjun78.github.io/portfolio/Apriori%20Association.html) · [Specialized ML Topics](https://wenjun78.github.io/portfolio/Specialized%20Machine%20Learning%20Topics.html) · [Big Data Longitudinal Analysis](https://wenjun78.github.io/portfolio/Big%20Data%20Longitudinal%20Analysis.html) · [Function Optimization](https://wenjun78.github.io/portfolio/Optimization.html) · [Deep Learning](https://wenjun78.github.io/portfolio/DeepLearning.html)-->

<!----Data Visualization Dashboards:
- **[Malaysian Car Market Analysis](https://app.powerbi.com/view?r=eyJrIjoiZGE1NTFmMTYtMjVjNy00NzAxLTgzYzktNzY3YjkyNDVlY2I3IiwidCI6ImE5ODIzYTlkLWJjYTItNDY5ZC1iNGZmLTJlMmIyZDE3NzY5MiIsImMiOjEwfQ%3D%3D)** (Power BI)
- **[Healthcare Analytics](https://public.tableau.com/views/HealthcareAnalytics_17015232280100/Dashboard1?:language=en-US&:display_count=n&:origin=viz_share_link)** (Tableau)
- **[Job Sphere Analytics](https://public.tableau.com/views/TableauinDataJobSphereAnAnalyticalDashboard/JobAnalytics?:language=en-US&:display_count=n&:origin=viz_share_link)** (Tableau)
<!---- **[Employer Satisfaction Survey 2017](https://tinyurl.com/2017-ess)** (Power BI)-->
<!---- **[Graduate Employment Study 2017](https://tinyurl.com/2017-sls)** (Power BI)-->
<!---- **[Analysis of Sales & Profit of Superstore Ltd](https://tinyurl.com/dynamiklau)**-->
<!--- **[Insight into the inpatient services of Hospital (2020)](https://tinyurl.com/laujpmc23)**-->
<!---- **[Airline Satisfaction Study](https://tinyurl.com/flightsatrat)**-->

<!---App:
- **[R Shiny App](https://f7cnn8-wj-lau.shinyapps.io/Assign5/)**-->



## Let's Connect :handshake:

Always open to collaboration, feedback, and interesting problems in AI and data.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/wjun/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/wenjun78)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wenjun78)
