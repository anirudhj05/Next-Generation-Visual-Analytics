# <img src="data/NextGen.png" alt="NextGen Analytics Logo" width="60"/> Next Generation Visual Analytics Dashboard
An interactive, user-friendly biomedical data dashboard designed to automate complex data analysis and visualization tasks for biomedical research. This dashboard integrates data preprocessing, exploratory analysis, visualization, dimensionality reduction, machine learning modeling, image segmentation, and a conversational AI assistant.

## About The Dashboard:

Next Generation Visual Analytics is a comprehensive Streamlit application that accelerates biomedical research by combining:
- Data Preprocessing: Handles missing values, detects outliers, removes duplicates, and includes logarithmic transformations.
- Exploratory Data Analysis: Includes quick previews, summary statistics, histograms, and box plots.
- Visualization: Includes scatter, line, bar, violin, pie, heatmap, 3D plots with a wide range of colour palettes.
- Dimensionality Reduction: Includes PCA, t-SNE, UMAP with parameter tuning, explained variance, feature loadings, and 2D/3D projections.
- Machine Learning Analysis: Includes supervised classification (Logistic Regression, Random Forest, SVM, XGBoost, KNN) with feature selection, stratified cross-validation, performance metrics, and model saving & inference.
- Medical Image Analysis: Includes NIfTI volumetric viewer, windowing, LungMask & TotalSegmentator organ segmentation, and overlay & download.
- Natural Image Analysis: Includes Grayscale, channel separation, blur, edge detection, thresholding, histogram equalization, and face detection.
- AI Assistant: Chat interface powered by Llama 4 Maverick with Replicate Llama 3 fallback, pre-loaded with dataset context for tailored guidance.

## File Structure

```bash
.
├── main.py                    # Main Streamlit dashboard
├── requirements.txt           # Python package dependencies
├── ai_assistant.py            # AI assistant module
├── dataset_analyzer.py        # AI assistant data analysis module
├── README.md                  # This file
├── data/                      # Includes sample datasets and test images for image analysis
│   └── NextGen.png            # Dashboard logo
├── Demo/                      # Demo Videos explaining how to use the dashboard                    
```

## Requirements
Make sure you have the following for the dashboard to work:
- OS: Linux (Ubuntu 20.04+), macOS (Catalina+), or Windows 10+
- Python: 3.8 ≤ version < 3.12
- RAM: ≥ 8 GB (16 GB recommended for large imaging tasks)
- Storage: ≥ 2 GB free for models & temp files
- Optional GPU: CUDA‑enabled GPU accelerates `totalsegmentator` inference
- Network: Internet access for AI Assistant API calls

## Installation

To reproduce this dashboard, execute the following steps:

1. Clone the repository
```bash
git clone https://github.com/<your-username>/next-generation-visual-analytics.git
cd next-generation-visual-analytics
```
2. Create a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.\.venv\\Scripts\\activate     # Windows PowerShell
```
3. Install Dependencies given in the `requirements.txt` file
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. Running the Dashboard

From the project root, run:
```bash
streamlit run main.py
```
### Use the Upload panel in the sidebar to load:
- Data: .csv, .tsv, .xls, .xlsx
- Medical Images: .nii or .nii.gz
- Natural Images: .jpg, .jpeg, .png
- Navigate tasks via the sidebar menu icons.

For more instructions, look at the demo videos provided in the `Demo` folder.

## Key Functionalities of Next Generation Visual Analytics

1. Data Preview & Summary:
Once the dataset is uploaded, users will be able to view the first N rows, the data types overview, row/column counts, histogram, and  box plot per numeric column.

2. Clean Data:
Users will be able to select missing-value strategies where they can choose between drop rows/columns with NaN, or fill with 0/mean/median/mode values. There is also an option to remove duplicate values in the duplicates tab, detect outliers (IQR method) and choose remove/cap, transform the columns (min-max, Z-score, log, square root), and finally download the cleaned data as a csv or excel file.

3. Filter & Plot:
Once the user has cleaned the dataset, there is a tab to plot figures which includes the following chart types:
- Scatter
- Line
- Bubble
- Bar (Grouped)
- Histogram
- Box
- Violin
- Pie
- Heatmap

Users can customize axes, color by categorical column, choose from a wide range of colour palettes (default, colorblind, pastel, bold, high contrast), and finally export the interactive Plotly figures or static PNG via Streamlit snapshot.

4. Dimensionality reduction:
This dashboard also has the option to perform dimensionality reduction once the dataset has been cleaned. The 3 most common dimensionality reduction methods (PCA, t-SNE, and UMAP) have been included in this dashboard.

  Users have the option of selecting all the numeric features or choose which specific features they want, along with optional standardization (recommended). 
  Each dimensionality reduction method has its own set of hyperparameters with default values, which can also be adjusted to the user's preference.
  ### Choose PCA/t-SNE/UMAP:
    - PCA: Set n_components; view explained variance bar+line, feature loadings table & heatmap, and plot 2D/3D scatter.
    - t-SNE: Adjust perplexity & n_iter; plot 2D scatter.
    - UMAP: Adjust n_neighbors & min_dist; plot 2D scatter.
  - Optionally append reduced dimensions to dataset and download updated CSV and plot as a PNG.

5. Machine Learning:
  The dashboard also includes ML modelling. The user gets to select the target label, and the dashboard automatically plots class distribution. Within the dashboard, the users can also manually select features or use the automated feature selection techniques based on statistical criteria, such as ANOVA F-value scoring or Mutual Information ranking. If the user selects automatic methods, the system generates a ranking of features based on their importance scores, prioritising the strongest predictors for the model. Following feature selection, the users have the option to apply feature preprocessing by standardising the features (zero mean and unit variance), which is recommended to ensure consistent feature scales, particularly for models sensitive to input magnitude, such as SVM and Logistic Regression.

To accommodate a range of predictive models, the dashboard allows users to choose from:  
- Logistic Regression
- Random Forest
- SVM
- XGBoost
- KNN

  Each model comes with customizable hyperparameters, enabling users to fine-tune the model settings based on the specific characteristics of their dataset. After configuring the model, users can define the number of folds for cross-validation. The system implements stratified k-fold cross-validation, which maintains the original class distribution within each fold to ensure fair evaluation. After training, the dashboard automatically evaluates model performance across the cross-validation folds. A summary table displaying fold-by-fold accuracy (and AUC for binary classification) is generated, providing a direct assessment of model stability across different splits. The average accuracy and AUC scores are displayed, offering users a quick overview of overall performance. To further interpret model behaviour, a confusion matrix is generated, visually presenting the true versus predicted class distributions. This allows the users to easily detect patterns such as high false positives or false negatives. To support reusability and practical deployment, the dashboard provides functionality to save trained models alongside their preprocessing pipeline and metadata, including selected features, label encoders, and normalization parameters. Models are exported as serialized joblib files, ensuring compatibility with standard Python environments. Additionally, users can upload a trained model and a new dataset to perform inference directly within the dashboard. The system validates whether required features are present in the new data, applies consistent preprocessing (e.g., scaling, encoding), and outputs predictions with optional downloading.
 
6. Medical Image Analysis: Users can  upload `.nii` or `.nii.gz` formatted CT or MRI scans. The dashnoard reads the volumetric data and displays axial slices with a slider for navigation. There are built-in controls for windowing (window center and width) to enable optimized visualization of anatomical structures and then perform segmentation. For organ segmentation, two powerful pre-trained models are available:

    (i) LungMask - Automatically segments lung regions using a deep learning-based model. The resulting lung mask is overlaid on the CT slices for inspection and can be downloaded as an image.
   
    (ii) TotalSegmentator - Provides fine-grained multi-organ segmentation. Users can choose from a wide range of anatomical structures (liver, spleen, kidneys, heart, lungs, aorta, pancreas, brain, bladder) and generate organ-specific masks. These masks are visualized in color overlays and are downloadable in .nii.gz format as well.

8. Natural Image Analysis: Users can upload `.jpg`, `.jpeg`, or `.png` images and apply a suite of computer vision techniques implemented using Opencv. The following functionalities are available:
- Grayscale conversion (transforms RGB images into single-channel grayscale)
- Channel separation (splits the image into Red, Green, and Blue channels for separate visualization)
- Image blurring and smoothing (includes Gaussian Blur, Median Blur, and Bilateral Filter options for noise reduction)
- Edge detection (including Canny, Sobel, and Laplacian operators to highlight boundaries and structures)
- Face detection (uses Haar cascades to locate and annotate human faces in the image)
- Thresholding (offers Binary, Binary Inverted, Truncated, To Zero, Adaptive Mean, Adaptive Gaussian, and Otsu's method to binarize images based on pixel intensity)
- Histogram Equalization (enhances image contrast, particularly useful in medical and low-light images)

   All processed images are displayed side-by-side with the original, and users can download the final outputs for further use. 

8. AI Assistant:
Finally, a conversational AI assistant is integrated into the dashboard to make it easier for biomedical researchers and doctors who lack computational skills to get started. This AI assistant is powered by Llama 4 Maverick with an option to fallback to Llama 3 if primary API gets rate limited. Once the user uploads the dataset,  the assistant automatically collects the following information so that it understands the data's structure and content before any questions are asked. 
(1) Statistical summary (means, variances, percentiles)
(2) Dataset structure (column names and data types)
(3) Preview of the first few rows

  After initialization, users can ask the assistant questions about any part of their analysis, such as understanding and interpreting PCA variance plots, tuning t-SNE perplexity, and selecting UMAP neighborhood sizes, as well as selecting suitable  ML algorithms and hyperparameters and providing suggestions on medical and natural image segmentation parameters. Under the hood, each user query is preceded by a system prompt incorporating the dataset information, ensuring that the answers are tailored to the actual variables and distributions. The assistant is rate-limited to 100 daily requests per user to maintain cost control while offering reliable service.

