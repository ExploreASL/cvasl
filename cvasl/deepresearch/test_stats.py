import argparse
import logging
import os
import re
import traceback
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import torch
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from torch.utils.data import DataLoader

from .data import BrainAgeDataset
from .utils import wrap_title

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BrainAgeAnalyzer:
    def __init__(self, validation_csv, validation_img_dir, mask_path, model_dir, output_root, use_cuda=False, demographics_columns=["Sex", "Site","Diagnosis"], group_columns=["Sex", "Site","Diagnosis"], indices_path = None): # Added group_columns as parameter
        self.validation_csv = validation_csv
        self.validation_img_dir = validation_img_dir
        self.model_dir = model_dir
        self.mask_path = mask_path
        self.output_root_base = output_root
        self.output_root = output_root
        self.group_cols = group_columns
        self.demographics_columns = demographics_columns
        self.use_cuda = use_cuda
        self.indices_path = indices_path
        os.makedirs(self.output_root, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        logging.info(f"Using device: {self.device}")
        if type(self.validation_csv) == str:
            self.validation_csv = [self.validation_csv]
        if type(self.validation_img_dir) == str:
            self.validation_img_dir = [self.validation_img_dir]
        self.validation_datasets = [BrainAgeDataset(_c,_v, indices=_i,cat_cols=["Sex", "Site", "Diagnosis"], mask_path=_m) for _c,_v,_i,_m in zip(self.validation_csv, self.validation_img_dir,self.indices_path,self.mask_path)]
        self.validation_dataset_names = [os.path.basename(_c).split(".")[0] for _c in self.validation_csv]
        logging.info(f"Loaded {len(self.validation_datasets)} validation datasets with shapes {[len(d) for d in self.validation_datasets]}")

    def load_model_from_name(self, model_path):
        """Loads a model based on its filename using load_model_with_params."""
        model_filename = os.path.basename(model_path)
        match = re.search(r'best__(.+?)_', model_filename)
        if match:
            model_type = match.group(1)
        else:
            model_type = 'unknown' 
        model = torch.load(model_path, map_location=self.device, weights_only = False)
        logging.info("Loaded model: %s of type %s", model_filename, model_type)
        return model, model_type
    
    def predict_ages(self, model, val_dataset):
        """Predicts ages for the validation dataset using the given model."""
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        all_predictions = []
        all_targets = []
        participant_ids = []
        all_demographics = [] 
        with torch.no_grad():
            for batch in val_loader:
                
                images = batch["image"].unsqueeze(1).to(self.device)
                ages = batch["age"].unsqueeze(1).to(self.device)
                demographics = batch["demographics"].cpu().numpy() 
                outputs = model(images, batch["demographics"].to(self.device))
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(ages.cpu().numpy().flatten())
                participant_ids.extend(batch["participant_id"])
                all_demographics.extend(demographics) 
        return participant_ids, all_predictions, all_targets, all_demographics 

    def analyze_heteroscedasticity(self, data, model_name, model_type, age_bins=None, alpha=0.05):
        """Analyzes heteroscedasticity of BAG across age bins using multiple tests and visualizations.

        Args:
            data (pd.DataFrame): DataFrame with 'actual_age', 'predicted_age', and 'brain_age_gap'.
            model_name (str): Name of the model.
            model_type (str): Type of the model.
            age_bins (np.ndarray, optional): Age bin edges. Defaults to np.linspace(0, 110, 12).
            alpha (float): Significance level for hypothesis tests. Defaults to 0.05.

        Returns:
            dict: Dictionary containing test results and binned data.
        """

        if data.empty:
            logging.warning(f"Input data is empty for {model_name}. Skipping analysis.")
            return {"results": None, "binned_data": None}

        if age_bins is None:
            age_bins = np.linspace(0, 110, 12)
            labels = [f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)]

            
            

        output_dir = self.output_root
        
        
        os.makedirs(output_dir, exist_ok=True)
        binned_data = data.copy()

        
        binned_data['age_bin_str'] = pd.cut(binned_data['actual_age'], bins=age_bins,
                                            labels=[f"{age_bins[i]:.0f}-{age_bins[i + 1]:.0f}" for i in
                                                    range(len(age_bins) - 1)], include_lowest=True, right=True)
        binned_data['age_bin'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=False, include_lowest=True,
                                        right=True)

        
        results = {}

        
        bag_groups = [group['brain_age_gap'].values for _, group in binned_data.groupby('age_bin')]
        bag_groups = [group for group in bag_groups if len(group) > 0]

        
        try:
            bp_lm, bp_p_value, _, _ = het_breuschpagan(binned_data['brain_age_gap'],
                                                      sm.add_constant(binned_data['actual_age']))
            results['breusch_pagan'] = {'statistic': bp_lm, 'p_value': bp_p_value}
        except Exception as e:
            logging.warning(f"Breusch-Pagan test failed for {model_name}: {e}")
            results['breusch_pagan'] = {'statistic': np.nan, 'p_value': np.nan}

        
        try:
            white_lm, white_p_value, _, _ = het_white(binned_data['brain_age_gap'],
                                                     sm.add_constant(binned_data['actual_age']))
            results['white'] = {'statistic': white_lm, 'p_value': white_p_value}
        except Exception as e:
            logging.warning(f"White test failed for {model_name}: {e}")
            results['white'] = {'statistic': np.nan, 'p_value': np.nan}

        title = (
            "Residuals (BAG) vs. Fitted Values (Predicted Age). This plot shows how the errors (residuals) "
            "are distributed across the range of predicted ages. Look for non-uniform spread: if the spread changes "
            "systematically (e.g., wider at higher ages), it indicates heteroscedasticity."
        )
        plt.figure(figsize=(8, 6))
        plt.scatter(data['predicted_age'], data['brain_age_gap'], alpha=0.5)
        plt.title(wrap_title(title), fontsize=9)
        plt.xlabel('Predicted Age')
        plt.ylabel('Residuals (BAG)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(os.path.join(output_dir, f"residuals_vs_fitted.png"))
        plt.close()

        standardized_residuals = (data['brain_age_gap'] - data['brain_age_gap'].mean()) / data['brain_age_gap'].std()
        sqrt_abs_standardized_residuals = np.sqrt(np.abs(standardized_residuals))

        title = (
            "Scale-Location Plot. This plot shows the spread of residuals (square root of absolute standardized residuals) "
            "against predicted age. A horizontal red line indicates constant variance (homoscedasticity). "
            "An upward or downward sloping line suggests heteroscedasticity."
        )
        plt.figure(figsize=(8, 6))
        plt.scatter(data['predicted_age'], sqrt_abs_standardized_residuals, alpha=0.5)
        plt.title(wrap_title(title), fontsize=9)
        plt.xlabel('Predicted Age')
        plt.ylabel('√|Standardized Residuals|')
        sns.regplot(x=data['predicted_age'], y=sqrt_abs_standardized_residuals, scatter=False, lowess=True, line_kws={'color': 'red'})
        plt.savefig(os.path.join(output_dir, f"scale_location.png"))
        plt.close()

        summary = []
        summary.append(f"Heteroscedasticity Analysis for {model_name} ({model_type}):")
        for test_name, test_result in results.items():
            if not np.isnan(test_result['p_value']):
                significant = test_result['p_value'] < alpha
                summary.append(
                    f"  - {test_name.replace('_', ' ').title()}: Statistic = {test_result['statistic']:.3f}, "
                    f"p-value = {test_result['p_value']:.3f} "
                    f"({'Significant' if significant else 'Not Significant'} at alpha={alpha})"
                )
            else:
                summary.append(f"  - {test_name.replace('_', ' ').title()}: Test failed or not applicable.")

        overall_conclusion = "Overall: Evidence of heteroscedasticity." if any(
            results[test]['p_value'] < alpha for test in results if
            not np.isnan(results[test]['p_value'])) else "Overall: No significant evidence of heteroscedasticity."
        summary.append(overall_conclusion)
        summary.append("Implications: If heteroscedasticity is present, consider transforming the data (e.g., log transform) or using weighted least squares regression.")


        return {"results": results, "binned_data": binned_data}

    def calculate_ccc(self, data, model_name, model_type):
        """
        Calculates and visualizes the Concordance Correlation Coefficient (CCC)
        between predicted and actual age, along with other relevant metrics and plots.
        """
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except:
                logging.error("Input 'data' must be a pandas DataFrame or convertible to one.")
                return np.nan
        if not {'actual_age', 'predicted_age'}.issubset(data.columns):
            logging.error("Input 'data' must contain 'actual_age' and 'predicted_age' columns.")
            return np.nan
        if not (pd.api.types.is_numeric_dtype(data['actual_age']) and pd.api.types.is_numeric_dtype(data['predicted_age'])):
            logging.error("'actual_age' and 'predicted_age' columns must contain numeric data.")
            return np.nan

        if data['actual_age'].isnull().any() or data['predicted_age'].isnull().any():
            logging.warning("NaN values found in 'actual_age' or 'predicted_age'. Removing rows with NaNs.")
            data = data.dropna(subset=['actual_age', 'predicted_age'])
        if len(data) == 0:
            logging.error("No valid data remaining after NaN removal.")
            return np.nan
        if data.empty:
            logging.error("'actual_age' and 'predicted_age' cannot be empty.")
            return np.nan


        long_data = pd.melt(data, value_vars=['actual_age', 'predicted_age'],
                            var_name='rater', value_name='age')
        long_data['subject'] = np.tile(np.arange(len(data)), 2) #repeats each index twice.
        icc_result = pg.intraclass_corr(data=long_data, targets='subject', raters='rater', ratings='age')

        ccc_value = icc_result[icc_result['Type'] == 'ICC3k']['ICC'].iloc[0]

        mae = np.mean(np.abs(data['predicted_age'] - data['actual_age']))
        rmse = np.sqrt(np.mean((data['predicted_age'] - data['actual_age'])**2))
        correlation = data['actual_age'].corr(data['predicted_age'])  # Pearson correlation

        output_path_txt = os.path.join(output_dir, f"metrics.txt")
        with open(output_path_txt, 'w') as f:
            f.write(f"CCC (Approximated by ICC3k): {ccc_value:.4f}\n")  # Clarify approximation
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Pearson Correlation: {correlation:.4f}\n")
        logging.info(f"Metrics saved to: {output_path_txt}")
        plt.figure(figsize=(8, 6))
        plt.scatter(data['actual_age'], data['predicted_age'], alpha=0.5)
        plt.xlabel("Actual Age")
        plt.ylabel("Predicted Age")
        m, b = np.polyfit(data['actual_age'], data['predicted_age'], 1)  # Linear regression
        plt.plot(data['actual_age'], m * data['actual_age'] + b, color='red', label=f'y = {m:.2f}x + {b:.2f}')

        min_val = min(data['actual_age'].min(), data['predicted_age'].min())
        max_val = max(data['actual_age'].max(), data['predicted_age'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Agreement')
        #============================================
        plt.title(wrap_title(f"Scatter Plot of Predicted vs. Actual Age\nThis plot shows the relationship between predicted and actual ages. Each point represents a subject.  The red line is the best-fit line, and the dashed black line represents perfect agreement (where predicted age equals actual age). Deviations from the dashed line indicate prediction errors.  A tighter clustering around the dashed line suggests better model performance."),fontsize=9)
        plt.legend()
        plt.tight_layout()
        scatter_plot_path = os.path.join(output_dir, f"scatter_plot.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        logging.info(f"Scatter plot saved to: {scatter_plot_path}")

        plt.figure(figsize=(8, 6))
        diff = data['predicted_age'] - data['actual_age']
        mean_age = (data['predicted_age'] + data['actual_age']) / 2
        plt.scatter(mean_age, diff, alpha=0.5)
        plt.axhline(np.mean(diff), color='red', linestyle='-', label=f'Mean Difference: {np.mean(diff):.2f}')
        plt.axhline(np.mean(diff) + 1.96*np.std(diff), color='red', linestyle='--', label=f'+1.96 SD: {np.mean(diff) + 1.96*np.std(diff):.2f}') #upper limit
        plt.axhline(np.mean(diff) - 1.96*np.std(diff), color='red', linestyle='--', label=f'-1.96 SD: {np.mean(diff) - 1.96*np.std(diff):.2f}') #lower limit
        plt.xlabel("Mean of Predicted and Actual Age")
        plt.ylabel("Difference (Predicted - Actual)")
        plt.title(wrap_title("Bland-Altman Plot of Age Prediction Differences\nThis plot shows the agreement between predicted and actual ages.  The x-axis represents the average of the predicted and actual ages, and the y-axis represents the difference between them. The solid red line shows the average difference (bias).  The dashed red lines represent the 95% limits of agreement (mean difference ± 1.96 times the standard deviation of the differences). Ideally, most points should fall within these limits, and the mean difference should be close to zero."),fontsize=9)
        plt.legend()
        plt.tight_layout()
        bland_altman_plot_path = os.path.join(output_dir, f"bland_altman_plot.png")
        plt.savefig(bland_altman_plot_path)
        plt.close()
        logging.info(f"Bland-Altman plot saved to: {bland_altman_plot_path}")

        return ccc_value

    def create_predictions_csv(self, participant_ids, predicted_ages, actual_ages, model_type, model_name):
        """Creates a CSV file with participant IDs, predicted ages, actual ages, and percentage error."""
        df = pd.DataFrame({
            "participant_id": participant_ids,
            "predicted_age": predicted_ages,
            "actual_age": actual_ages
        })
        df["brain_age_gap"] = df["predicted_age"] - df["actual_age"]
        df["percentage_error"] = (np.abs(df["brain_age_gap"]) / df["actual_age"]) * 100
        output_path = self.output_root
        #add predictions.csv to the output path
        output_path = os.path.join(self.output_root, f"predictions.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to: {output_path}")

    def calculate_iqr(self, series):
        """Calculates the interquartile range (IQR) for a given series."""
        if len(series) <= 1:
            return np.nan
        return series.quantile(0.75) - series.quantile(0.25)

    def calculate_descriptive_stats(self, data, group_cols=None):
        """
        Calculates descriptive statistics for the given data.

        Args:
            data (pd.DataFrame): DataFrame containing 'predicted_age', 'actual_age',
                                 'brain_age_gap', and 'participant_id' columns.
            group_cols (list or str, optional): Column(s) to group by. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing descriptive statistics.
        """

        for col in ["predicted_age", "actual_age", "brain_age_gap"]:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric.")

        if group_cols:
            stats_df = data.groupby(group_cols).agg(
                mean_predicted_age=("predicted_age", "mean"),
                std_predicted_age=("predicted_age", "std"),
                median_predicted_age=("predicted_age", "median"),
                mean_actual_age=("actual_age", "mean"),
                std_actual_age=("actual_age", "std"),
                median_actual_age=("actual_age", "median"),
                mean_bag=("brain_age_gap", "mean"),
                std_bag=("brain_age_gap", "std"),
                median_bag=("brain_age_gap", "median"),
                count=("participant_id", "count"),
                iqr_predicted_age=("predicted_age", self.calculate_iqr),
                iqr_actual_age=("actual_age", self.calculate_iqr),
                iqr_bag=("brain_age_gap", self.calculate_iqr),
            )

        else:
            stats_df = data.agg(
                mean_predicted_age=("predicted_age", "mean"),
                std_predicted_age=("predicted_age", "std"),
                median_predicted_age=("predicted_age", "median"),
                mean_actual_age=("actual_age", "mean"),
                std_actual_age=("actual_age", "std"),
                median_actual_age=("actual_age", "median"),
                mean_bag=("brain_age_gap", "mean"),
                std_bag=("brain_age_gap", "std"),
                median_bag=("brain_age_gap", "median"),
                count=("participant_id", "count"),
            ).T

            iqr_values = {
                "iqr_predicted_age": self.calculate_iqr(data["predicted_age"]),
                "iqr_actual_age": self.calculate_iqr(data["actual_age"]),
                "iqr_bag": self.calculate_iqr(data["brain_age_gap"]),
            }
            iqr_df = pd.DataFrame([iqr_values])

            stats_df = pd.concat([stats_df, iqr_df], axis=1)
        self.visualize_descriptive_stats(data, stats_df, group_cols=group_cols)
        return stats_df
    
    
    def visualize_descriptive_stats(self, data, stats_df, group_cols=None, output_dir=None):
        """
        Generates and saves visualizations based on descriptive statistics.

        Args:
            data (pd.DataFrame):  The original data DataFrame.
            stats_df (pd.DataFrame): DataFrame of descriptive statistics from calculate_descriptive_stats.
            group_cols (list or str, optional):  Grouping columns (must match calculate_descriptive_stats).
            output_dir (str): Directory to save the plots.  Creates if it doesn't exist.
        """
        
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)

        if group_cols:
            if isinstance(group_cols, str):
                group_cols = [group_cols] 

            for group_col in group_cols:
                plt.figure(figsize=(10, 6))
                
                data = data.reset_index(drop=True)
                sns.boxplot(x=group_col, y='brain_age_gap', data=data)
                plt.title(wrap_title(f"Brain Age Gap Distribution by {group_col}\nThis box plot shows the distribution of the brain age gap for each group defined by '{group_col}'.  Comparing the medians, interquartile ranges, and presence of outliers across groups can reveal differences in model performance between groups."), fontsize=9)
                plt.xlabel(group_col)
                plt.ylabel("Brain Age Gap")
                plt.tight_layout()
                boxplot_path = os.path.join(output_dir, f"brain_age_gap_boxplot_{group_col}.png")
                plt.savefig(boxplot_path)
                plt.close()
                logging.info(f"Box plot for {group_col} saved to: {boxplot_path}")

                plt.figure(figsize=(12, 6))
                melted_data = pd.melt(data, id_vars=group_col, value_vars=['predicted_age', 'actual_age'], var_name='Type', value_name='Age')
                sns.boxplot(x=group_col, y='Age', hue='Type', data=melted_data)
                plt.title(wrap_title(f"Predicted and Actual Age Distribution by {group_col}\nThis box plot shows the distributions of both predicted and actual ages for each group in '{group_col}'.  Comparing the distributions helps assess whether the model performs differently across groups and whether there are systematic biases within specific groups."), fontsize=9)
                plt.xlabel(group_col)
                plt.ylabel("Age")
                plt.tight_layout()
                pred_vs_actual_boxplot_path = os.path.join(output_dir, f"pred_vs_actual_boxplot_{group_col}.png")
                plt.savefig(pred_vs_actual_boxplot_path)
                plt.close()
                logging.info(f"Predicted vs Actual box plot for {group_col} saved to: {pred_vs_actual_boxplot_path}")



    def analyze_bag_by_age_bins(self, data, model_name, model_type, age_bins=np.linspace(0, 110, 12)):
        """Analyzes Brain Age Gap (BAG) within different age bins.

        Args:
            data (pd.DataFrame): DataFrame containing 'actual_age', 'predicted_age', and 'brain_age_gap' columns.
            model_name (str): Name of the model.  (Used for filenames, etc. - not directly in this function)
            model_type (str): Type of the model. (Used for filenames, etc. - not directly in this function)
            age_bins (array-like):  The age bins to use. Bins are right-inclusive (e.g., age 30 falls into the 30-40 bin).

        Returns:
            tuple: (bin_stats_df, binned_data)
                bin_stats_df (pd.DataFrame): DataFrame containing BAG statistics for each age bin.
                binned_data (pd.DataFrame):  The input 'data' DataFrame with an added 'age_bin' column.
        """
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)

        binned_data = data.copy()
        #age_labels = [f'{age_bins[i]}-{age_bins[i+1]}' for i in range(len(age_bins)-1)]
        binned_data['age_bin'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=False, include_lowest=True, right=True)

        bin_stats = []
        labels = [f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)]

        binned_data['age_bin'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=False, include_lowest=True, right=True)
        binned_data['age_bin_label'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=labels, include_lowest=True, right=True)

        for bin_label, bin_group in binned_data.groupby('age_bin'):
            bag_values = bin_group['brain_age_gap']
            actual_ages_bin = bin_group['actual_age']
            predicted_ages_bin = bin_group['predicted_age'] 

            epsilon = 1e-6 

            mape_bag = np.mean(np.abs(bag_values) / (actual_ages_bin + epsilon)) * 100 if not actual_ages_bin.empty else np.nan
            mdape_bag = np.median(np.abs(bag_values) / (actual_ages_bin + epsilon)) * 100 if not actual_ages_bin.empty else np.nan
            rmse_bag = np.sqrt(mean_squared_error([0] * len(bag_values), bag_values))

            bin_stat = {
                'age_bin_label': f"{age_bins[int(bin_label)]:.1f}-{age_bins[int(bin_label)+1]:.1f}" if bin_label < len(age_bins)-1 else f">={age_bins[int(bin_label)]:.1f}",
                'mean_bag': bag_values.mean(),
                'std_bag': bag_values.std(),
                'variance_bag': bag_values.var(),
                'median_bag': bag_values.median(),
                'iqr_bag': np.percentile(bag_values, 75) - np.percentile(bag_values, 25),
                'mape_bag': mape_bag,
                'mdape_bag': mdape_bag,
                'rmse_bag': rmse_bag,
                'count': bag_values.count()
            }
            bin_stats.append(bin_stat)

        bin_stats_df = pd.DataFrame(bin_stats)
        self.visualize_bag_analysis(bin_stats_df, binned_data, model_name)
        return bin_stats_df, binned_data



    def visualize_bag_analysis(self, bin_stats_df, binned_data, model_name):
        """
        Visualizes the results of the BAG analysis by age bin.

        Args:
            bin_stats_df (pd.DataFrame): Output DataFrame from analyze_bag_by_age_bins (bin statistics).
            binned_data (pd.DataFrame): Output DataFrame from analyze_bag_by_age_bins (data with age bins).
            model_name (str):  The name of the model (used for file naming -  not directly in the plots).
        """

        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({'font.size': 9})

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        binned_data = binned_data.reset_index(drop=True)
        sns.boxplot(x='age_bin_label', y='brain_age_gap', data=binned_data, showmeans=True,
                    meanprops={"markerfacecolor": "red", "markeredgecolor": "black"})
        plt.title(wrap_title("Boxplot of Brain Age Gap (BAG) by Age Bin\nShows the distribution of BAG within each age group.  The box represents the interquartile range (IQR), the line is the median, and whiskers extend to 1.5*IQR.  Outliers are shown as individual points. The red triangle represents mean."),fontsize=9)
        plt.xlabel("Age Bin")
        plt.ylabel("Brain Age Gap (Years)")

        plt.subplot(1, 2, 2)
        sns.violinplot(x='age_bin_label', y='brain_age_gap', data=binned_data, inner="quartile")
        plt.title(wrap_title("Violin Plot of Brain Age Gap (BAG) by Age Bin\nDisplays the distribution of BAG, showing the density of data points at different BAG values.  Wider sections indicate higher density. Lines represent the quartiles (25th, 50th, 75th percentiles) of the data within each bin."), fontsize=9)
        plt.xlabel("Age Bin")
        plt.ylabel("Brain Age Gap (Years)")

        plt.tight_layout()
        bag_dist_path = os.path.join(output_dir, f"bag_distribution_by_age_bin.png")
        plt.savefig(bag_dist_path)
        plt.close()
        logging.info(f"BAG distribution plots saved to: {bag_dist_path}")


        plt.figure(figsize=(10, 6))
        bar_width = 0.35

        bin_stats_df['age_bin_label'] = bin_stats_df['age_bin_label'].astype(str)

        x = np.arange(len(bin_stats_df['age_bin_label']))

        plt.bar(x - bar_width/2, bin_stats_df['mean_bag'], bar_width, label='Mean BAG', color='skyblue')
        plt.bar(x + bar_width/2, bin_stats_df['median_bag'], bar_width, label='Median BAG', color='lightcoral')
        plt.xticks(x, bin_stats_df['age_bin_label'])
        plt.title(wrap_title("Mean and Median Brain Age Gap (BAG) by Age Bin\nCompares the mean (average) and median BAG for each age group.  Differences between the mean and median can highlight the presence of outliers or skewed distributions."), fontsize=9)
        plt.xlabel("Age Bin")
        plt.ylabel("Brain Age Gap (Years)")
        plt.legend()
        plt.tight_layout()
        mean_median_bag_path = os.path.join(output_dir, f"mean_median_bag_by_age_bin.png")
        plt.savefig(mean_median_bag_path)
        plt.close()
        logging.info(f"Mean/Median BAG plot saved to: {mean_median_bag_path}")

        plt.figure(figsize=(10, 6))
        x = np.arange(len(bin_stats_df['age_bin_label']))

        plt.bar(x - bar_width, bin_stats_df['mape_bag'], bar_width, label='MAPE', color='mediumseagreen')
        plt.bar(x, bin_stats_df['mdape_bag'], bar_width, label='MdAPE', color='gold')
        plt.bar(x + bar_width, bin_stats_df['rmse_bag'], bar_width, label='RMSE', color='tomato')

        plt.xticks(x, bin_stats_df['age_bin_label'])
        plt.title(wrap_title("Error Metrics (MAPE, MdAPE, RMSE) by Age Bin\nShows different error measures for each age group. MAPE and MdAPE represent percentage errors, while RMSE is in the original unit (years).  Lower values indicate better performance."),fontsize=9)
        plt.xlabel("Age Bin")
        plt.ylabel("Error Value")
        plt.legend()
        plt.tight_layout()
        error_metrics_path = os.path.join(output_dir, f"error_metrics_by_age_bin.png")
        plt.savefig(error_metrics_path)
        plt.close()
        logging.info(f"Error metrics plot saved to: {error_metrics_path}")

        plt.figure(figsize=(8, 5))
        plt.bar(bin_stats_df['age_bin_label'], bin_stats_df['count'], color='lightslategray')
        plt.title(wrap_title("Number of Subjects per Age Bin\nDisplays the number of subjects in each age group.  This helps assess the reliability of statistics in each bin; larger sample sizes generally lead to more reliable results."), fontsize=9)
        plt.xlabel("Age Bin")
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        sample_size_path = os.path.join(output_dir, f"sample_size_by_age_bin.png")
        plt.savefig(sample_size_path)
        plt.close()
        logging.info(f"Sample size plot saved to: {sample_size_path}")

        plt.figure(figsize=(8, 6))
        sns.regplot(x='actual_age', y='brain_age_gap', data=binned_data, scatter_kws={'alpha':0.3}, line_kws={"color": "red"})
        plt.title(wrap_title("Brain Age Gap (BAG) vs. Actual Age\nShows the relationship between BAG and actual age.  Each point is a subject.  The red line is a trendline, indicating the general tendency of BAG across ages. Ideally, BAG should be centered around zero across all ages."), fontsize=9)
        plt.xlabel("Actual Age")
        plt.ylabel("Brain Age Gap (Years)")
        plt.axhline(0, color='black', linestyle='--')  # Add a horizontal line at BAG=0
        plt.tight_layout()
        bag_vs_age_path = os.path.join(output_dir, f"bag_vs_actual_age.png")
        plt.savefig(bag_vs_age_path)
        plt.close()
        logging.info(f"BAG vs. Actual Age plot saved to: {bag_vs_age_path}")

        num_bins = len(bin_stats_df['age_bin_label'])
        rows = int(np.ceil(np.sqrt(num_bins)))
        cols = int(np.ceil(num_bins / rows))
        plt.figure(figsize=(15, 15))

        for i, bin_label in enumerate(bin_stats_df['age_bin_label']):
          bin_data = binned_data[binned_data['age_bin'] == i]
          min_val = min(bin_data['actual_age'].min(), bin_data['predicted_age'].min())
          max_val = max(bin_data['actual_age'].max(), bin_data['predicted_age'].max())

          plt.subplot(rows, cols, i + 1)
          plt.scatter(bin_data['actual_age'], bin_data['predicted_age'], alpha=0.5)
          plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Agreement')
          plt.title(f"Bin: {bin_label}")
          plt.xlabel("Actual Age")
          plt.ylabel("Predicted Age")
          plt.legend()


        plt.suptitle(wrap_title("Predicted vs. Actual Age within Each Age Bin\nEach subplot shows the predicted vs. actual age for a specific age bin.  The dashed line represents perfect agreement.  This allows for a bin-specific assessment of prediction accuracy."), fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pred_vs_actual_per_bin_path = os.path.join(output_dir, f"predicted_vs_actual_age_per_bin.png")
        plt.savefig(pred_vs_actual_per_bin_path)
        plt.close()
        logging.info(f"Predicted vs. Actual Age per Bin plot saved to: {pred_vs_actual_per_bin_path}")
        
    def analyze_bias_variance_vs_age(self, data, model_name, model_type, age_bins=np.linspace(0, 110, 12)): # Added age_bins parameter
        """Analyzes bias and variance of predicted age across actual age distribution.

        Args:
            data (pd.DataFrame): DataFrame containing the following columns:
                - 'actual_age':  The true age of each subject.
                - 'predicted_age': The age predicted by the model.
                - 'brain_age_gap': The difference between predicted and actual age (predicted_age - actual_age).
            model_name (str): Name of the model (used for file naming, etc. - not directly used here).
            model_type (str): Type of the model (used for file naming, etc. - not directly used here).
            age_bins (np.ndarray):  Array defining the age bin edges.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an age bin, and the columns
                contain bias, variance, and error metrics for that bin.  The columns are:
                - 'age_bin_label':  String representing the age range of the bin (e.g., "20.0-30.0").
                - 'bias':  The mean Brain Age Gap (BAG) within the bin.
                - 'variance_bag': The variance of the BAG within the bin.
                - 'variance_predicted_age': The variance of the predicted ages within the bin.
                - 'mape_predicted_age': Mean Absolute Percentage Error of predicted age within the bin.
                - 'mdape_predicted_age': Median Absolute Percentage Error of predicted age within the bin.
                - 'rmse_predicted_age': Root Mean Squared Error of predicted age within the bin.
                - 'count': The number of samples within the bin.
        """
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)
        labels = [f"{int(age_bins[i])}-{int(age_bins[i+1])}" for i in range(len(age_bins)-1)]

        binned_data = data.copy()
        binned_data['age_bin'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=False, include_lowest=True, right=True)
        binned_data['age_bin_label'] = pd.cut(binned_data['actual_age'], bins=age_bins, labels=labels, include_lowest=True, right=True)


        bias_variance_stats = []
        for bin_label, bin_group in binned_data.groupby('age_bin'):
            bag_values_bin = bin_group['brain_age_gap']
            predicted_age_values = bin_group['predicted_age']
            actual_ages_bin = bin_group['actual_age'] 


            mape_predicted_age = np.mean(np.abs(predicted_age_values - actual_ages_bin) / actual_ages_bin) * 100 if not actual_ages_bin.empty else np.nan 
            mdape_predicted_age = np.median(np.abs(predicted_age_values - actual_ages_bin) / actual_ages_bin) * 100 if not actual_ages_bin.empty else np.nan 
            rmse_predicted_age = np.sqrt(mean_squared_error(actual_ages_bin, predicted_age_values)) 


            bin_stat = {
                'age_bin_label': f"{age_bins[int(bin_label)]:.1f}-{age_bins[int(bin_label)+1]:.1f}" if bin_label < len(age_bins)-1 else f">={age_bins[int(bin_label)]:.1f}", # Adjusted bin label
                'bias': bag_values_bin.mean(), 
                'variance_bag': bag_values_bin.var(), 
                'variance_predicted_age': predicted_age_values.var(), 
                'mape_predicted_age': mape_predicted_age, 
                'mdape_predicted_age': mdape_predicted_age, 
                'rmse_predicted_age': rmse_predicted_age, 
                'count': bag_values_bin.count()
            }
            bias_variance_stats.append(bin_stat)
        bias_variance_df = pd.DataFrame(bias_variance_stats)
        self.visualize_bias_variance(bias_variance_df, data)
        return bias_variance_df

    def visualize_bias_variance(self, bias_variance_df, data):
        """
        Visualizes the bias-variance analysis results with detailed plots.

        Args:
            bias_variance_df (pd.DataFrame): The output DataFrame from analyze_bias_variance_vs_age.
            data (pd.DataFrame): The original DataFrame used for the analysis, containing
                'actual_age', 'predicted_age', and 'brain_age_gap' columns.
        """

        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.histplot(data['brain_age_gap'], kde=True, color='purple')
        plt.title(wrap_title("Distribution of Brain Age Gap (BAG)\nThis histogram shows the distribution of the Brain Age Gap (predicted age - actual age).  A symmetrical distribution centered around zero suggests unbiased predictions.  The curve (Kernel Density Estimate) provides a smoothed representation of the distribution."),fontsize=9)
        plt.xlabel("Brain Age Gap")
        plt.ylabel("Frequency")
        plt.tight_layout()
        bag_distribution_plot_path = os.path.join(output_dir, "bag_distribution.png")
        plt.savefig(bag_distribution_plot_path)
        plt.close()
        logging.info(f"BAG distribution plot saved to: {bag_distribution_plot_path}")            

    @staticmethod
    def cohen_d(group1, group2):
        """Calculates Cohen's d for two groups."""
        diff = group1.mean() - group2.mean()
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return diff / np.sqrt(pooled_var + 1e-9) # Added epsilon to denominator


    def calculate_effect_sizes(self, data, group_cols, reference_group=None):
        """
        Calculates Cohen's d for predicted age and BAG between groups.

        Args:
            data: DataFrame with predicted_age, BAG, and group columns.
            group_cols: List of columns defining the groups.
            reference_group: Optional. The name of the reference group for comparisons.
                             If None, all pairwise comparisons are made.

        Returns:
            DataFrame with effect sizes.
        """

        results = []

        unique_groups = data.groupby(group_cols).groups.keys()

        if reference_group:
            comparisons = [
                (reference_group, group) for group in unique_groups if group != reference_group
            ]
        else:
            comparisons = list(combinations(unique_groups, 2))

        for group1_keys, group2_keys in comparisons:
            group1 = data.loc[
                data[group_cols].apply(
                    lambda row: tuple(row.values) == group1_keys, axis=1
                ),
                "predicted_age",
            ]
            group2 = data.loc[
                data[group_cols].apply(
                    lambda row: tuple(row.values) == group2_keys, axis=1
                ),
                "predicted_age",
            ]

            group1_bag = data.loc[
                data[group_cols].apply(
                    lambda row: tuple(row.values) == group1_keys, axis=1
                ),
                "brain_age_gap",
            ]  # Corrected column name to brain_age_gap
            group2_bag = data.loc[
                data[group_cols].apply(
                    lambda row: tuple(row.values) == group2_keys, axis=1
                ),
                "brain_age_gap",
            ]  # Corrected column name to brain_age_gap

            d_predicted_age = self.cohen_d(group1, group2)
            d_bag = self.cohen_d(group1_bag, group2_bag)

            results.append(
                {
                    "group_comparison": f"{group1_keys} vs {group2_keys}",
                    "cohen_d_predicted_age": d_predicted_age,
                    "cohen_d_bag": d_bag,
                }
            )
        results = pd.DataFrame(results)
        self.visualize_effect_sizes(results, data, group_cols)
        return results
    
    def visualize_effect_sizes(self, effect_sizes_df, data, group_cols):
        """
        Visualizes and analyzes the effect sizes calculated by calculate_effect_sizes.

        Args:
            effect_sizes_df: DataFrame returned by calculate_effect_sizes.
            data: The original DataFrame used for calculations.
            group_cols: The grouping columns.

        """
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)
        # 3. Distribution Plots (Histograms and KDE)
        for metric in ["predicted_age", "brain_age_gap"]:
            plt.figure(figsize=(10, 6))
            for group_keys in data.groupby(group_cols).groups.keys():
                group_data = data.loc[
                    data[group_cols].apply(
                        lambda row: tuple(row.values) == group_keys, axis=1
                    ),
                    metric,
                ]
                sns.histplot(
                    group_data,
                    kde=True,
                    label=str(group_keys[1]),
                    stat="density",
                    element="step",
                    bins=30,
                )  # Use density for better comparison
                

            title_text = (
                f"Distribution of {metric} Across Groups\n"
                f"This plot shows the distribution of {metric} for each group using histograms and kernel density estimates (KDEs). "
                f"Overlapping distributions indicate similarity, while distinct distributions suggest differences between groups."
            )
            plt.title(wrap_title(title_text), fontsize=9)
            plt.xlabel(metric)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            dist_plot_path = os.path.join(output_dir, f"distribution_{metric}.png")
            plt.savefig(dist_plot_path)
            plt.close()
            logging.info(f"Distribution plot for {metric} saved to: {dist_plot_path}")

    def plot_qq_plots(self, data, model_name, model_type):
        """Creates Q-Q plots for predicted age vs. actual age."""
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True)


        # Calculate quantiles
        theoretical_quantiles = np.linspace(0, 1, len(data))
        predicted_age_quantiles = np.quantile(data["predicted_age"], theoretical_quantiles)
        actual_age_quantiles = np.quantile(data["actual_age"], theoretical_quantiles)

        # Q-Q plot
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_age_quantiles, predicted_age_quantiles, alpha=0.7)
        min_val = min(np.min(actual_age_quantiles), np.min(predicted_age_quantiles))
        max_val = max(np.max(actual_age_quantiles), np.max(predicted_age_quantiles))
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--") # Identity line
        plt.title(f"{model_name} - Q-Q Plot (Predicted vs. Actual Age)")
        plt.xlabel("Actual Age Quantiles")
        plt.ylabel("Predicted Age Quantiles")
        plt.grid(True)
        output_path = os.path.join(output_dir, f"qq_plot.png")
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Q-Q plot saved to: {output_path}")

    def run_all_analyses(self):
        """Loads models, predicts ages, and performs all analyses."""
        
        for model_file in os.listdir(self.model_dir):
            logging.info(f"Number of datasets: {len(self.validation_datasets)}")            
            
            if model_file.endswith(".pth"):
                model_path = os.path.join(self.model_dir, model_file)
                try:
                    model, model_type = self.load_model_from_name(model_path)
                    logging.info(f"Loaded model: {model_file}")
                    predictions_df_all = []
                    demographics_df_all = []
                    
                    
                    for val_dataset, val_dataset_name in zip(self.validation_datasets, self.validation_dataset_names):
                        #get validation dataset class name
                        
                        self.output_root = os.path.join(self.output_root_base, model_file.split('.pth')[0], val_dataset_name)
                        os.makedirs(self.output_root, exist_ok=True)

                        print(val_dataset)
                        participant_ids, predicted_ages, actual_ages, demographics_list = self.predict_ages(model, val_dataset) # Get demographics
                        if not participant_ids:
                            logging.warning(f"Skipping model {model_file} due to empty participant_ids after prediction.")
                            continue
                        logging.info(f"Predictions made for model: {model_file}")
                        # Create the DataFrame directly here, ensuring 'participant_id' is included
                        predictions_df = pd.DataFrame({
                            "participant_id": participant_ids,
                            "predicted_age": predicted_ages,
                            "actual_age": actual_ages,
                            "brain_age_gap": np.array(predicted_ages) - np.array(actual_ages)  # Calculate BAG here
                        })
                        
                        self.create_predictions_csv(participant_ids, predicted_ages, actual_ages, model_file, model_type)

                        # Convert demographics list to DataFrame and concatenate
                        print("================================$$$$$$$$$$$$$$$$$$",demographics_list,self.demographics_columns)
                        demographics_df = pd.DataFrame(np.array(demographics_list), columns=self.demographics_columns) # Create demographics DF
                        #for any of the group cols, if it is not the demographics_df, add them to the demographics_df from val_dataset making sure the participant_ids match. participant_id is string. be very careful. some of the group columns might already be there
                        for col in self.group_cols:
                            demographics_df[col] = val_dataset.original_data_df[col]
                            demographics_df[col] = demographics_df[col].fillna('NA')
                        predictions_df = pd.concat([predictions_df, demographics_df], axis=1) # Concatenate demographics
                        
                        logging.info(f"Demographics added to predictions_df for model: {model_file}")

                        predictions_df_all.append(predictions_df)
                        demographics_df_all.append(demographics_df)
                        # Descriptive Statistics
                        logging.info(f"Running descriptive statistics for model: {model_file}")
                        descriptive_stats_df = self.calculate_descriptive_stats(predictions_df)
                        
                        logging.info(f"Descriptive statistics saved to: {self.output_root}")
                        # Descriptive Statistics by Group
                        if set(self.group_cols).issubset(predictions_df.columns): # Use self.group_cols
                            logging.info(f"Running descriptive statistics by group for model: {model_file}")
                            descriptive_stats_by_group_df = self.calculate_descriptive_stats(predictions_df, group_cols=self.group_cols) # Use self.group_cols
                            
                            logging.info(f"Descriptive statistics by group saved to: {self.output_root}")
                            # Calculate effect sizes between groups
                            logging.info(f"Calculating effect sizes for model: {model_file}")
                            effect_sizes_df = self.calculate_effect_sizes(predictions_df, group_cols=self.group_cols) # Use self.group_cols
                            
                            logging.info(f"Effect sizes saved to: {self.output_root}")
                        else:
                            logging.warning("Skipping descriptive statistics by group and effect size calculation - required columns not found.")

                        # Age Bin Analysis for BAG
                        logging.info(f"Running BAG analysis by age bin for model: {model_file}")
                        bag_by_age_bin_df = self.analyze_bag_by_age_bins(predictions_df, model_file, model_type) # Call new function
                        # Q-Q Plots
                        logging.info(f"Creating Q-Q plots for model: {model_file}")
                        self.plot_qq_plots(predictions_df, model_file, model_type)
                        # ICC Analysis # Add this section here, after QQ plots for example
                        logging.info(f"Running CCC analysis for model: {model_file}")
                        ccc_value = self.calculate_ccc(predictions_df, model_file, model_type)
                        logging.info(f"CCC Value for {model_file}: {ccc_value:.4f}")
                        # Heteroscedasticity Analysis # Add this section after CCC analysis
                        logging.info(f"Running heteroscedasticity analysis for model: {model_file}")
                        levene_stat, levene_p_value = self.analyze_heteroscedasticity(predictions_df, model_file, model_type)
                        # Bias and Variance vs Age Analysis
                        logging.info(f"Running bias and variance vs age analysis for model: {model_file}")
                        bias_variance_df = self.analyze_bias_variance_vs_age(predictions_df, model_file, model_type) # Call new fF_oriunction
                        # Metrics vs Age Plots (Example: MAE vs Age)

                    for _p,_s in zip(predictions_df_all, self.validation_dataset_names):
                        _p['Site'] = _s            
                    
                    self.output_root = os.path.join(self.output_root_base, model_file.split('.p')[0])
                    os.makedirs(self.output_root, exist_ok=True)
                    predictions_df = pd.concat(predictions_df_all)
                    demographics_df = pd.concat(demographics_df_all)
                    

                    # Descriptive Statistics
                    logging.info(f"Running descriptive statistics for model: {model_file}")
                    try:
                        descriptive_stats_df = self.calculate_descriptive_stats(predictions_df)
                        
                        logging.info(f"Descriptive statistics saved to: {self.output_root}")
                    except Exception as e:
                        logging.error(f"Error calculating descriptive statistics for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")
                    # Descriptive Statistics by Group
                    if set(self.group_cols).issubset(predictions_df.columns): # Use self.group_cols
                        try:
                            logging.info(f"Running descriptive statistics by group for model: {model_file}")
                            descriptive_stats_by_group_df = self.calculate_descriptive_stats(predictions_df, group_cols=self.group_cols) # Use self.group_cols
                            
                            logging.info(f"Descriptive statistics by group saved to: {self.output_root}")
                        except Exception as e:
                            logging.error(f"Error calculating descriptive statistics by group for model {model_file}:", exc_info=True)
                            tb_str = traceback.format_exc()
                            logging.error(f"{e}\nTraceback:\n{tb_str}")
                        
                        try:
                        # Calculate effect sizes between groups
                            logging.info(f"Calculating effect sizes for model: {model_file}")
                            effect_sizes_df = self.calculate_effect_sizes(predictions_df, group_cols=self.group_cols) # Use self.group_cols
                            
                            logging.info(f"Effect sizes saved to: {self.output_root}")
                        except Exception as e:
                            logging.error(f"Error calculating effect sizes for model {model_file}:", exc_info=True)
                            tb_str = traceback.format_exc()
                            logging.error(f"{e}\nTraceback:\n{tb_str}")
                    else:
                        logging.warning("Skipping descriptive statistics by group and effect size calculation - required columns not found.")

                    # Age Bin Analysis for BAG
                    try:
                        logging.info(f"Running BAG analysis by age bin for model: {model_file}")
                        bag_by_age_bin_df = self.analyze_bag_by_age_bins(predictions_df, model_file, model_type) # Call new function
                    except Exception as e:
                        logging.error(f"Error running BAG analysis by age bin for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")
                    try:
                        # Q-Q Plots
                        logging.info(f"Creating Q-Q plots for model: {model_file}")
                        self.plot_qq_plots(predictions_df, model_file, model_type)
                    except Exception as e:
                        logging.error(f"Error creating Q-Q plots for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")
                    try:
                        # ICC Analysis # Add this section here, after QQ plots for example
                        logging.info(f"Running CCC analysis for model: {model_file}")
                        ccc_value = self.calculate_ccc(predictions_df, model_file, model_type)
                    except Exception as e:
                        logging.error(f"Error running CCC analysis for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")
                    
                    try:
                        logging.info(f"Running heteroscedasticity analysis for model: {model_file}")
                        levene_stat, levene_p_value = self.analyze_heteroscedasticity(predictions_df, model_file, model_type)
                    except Exception as e:
                        logging.error(f"Error running heteroscedasticity analysis for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")
                    try:
                        # Bias and Variance vs Age Analysis
                        logging.info(f"Running bias and variance vs age analysis for model: {model_file}")
                        bias_variance_df = self.analyze_bias_variance_vs_age(predictions_df, model_file, model_type) # Call new fF_oriunction
                    except Exception as e:
                        logging.error(f"Error running bias and variance vs age analysis for model {model_file}:", exc_info=True)
                        tb_str = traceback.format_exc()
                        logging.error(f"{e}\nTraceback:\n{tb_str}")                    
                except Exception as e:
                    logging.error(f"Error processing model {model_file}:", exc_info=True)
                    tb_str = traceback.format_exc()
                    logging.error(f"{e}\nTraceback:\n{tb_str}")
                    return None, None
                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Age Analysis Script")
    parser.add_argument("--validation_csv", type=str, nargs='+', default="/home/radv/samiri/my-scratch/trainingdata/masked/topmri.csv", help="Path to the training CSV file")
    parser.add_argument("--validation_img_dir", type=str, nargs='+', default="/home/radv/samiri/my-scratch/trainingdata/masked/topmri/", help="Path to the training image directory")
    parser.add_argument("--model_dir", type=str, default="./saved_models", help="Path to the directory containing saved models")
    parser.add_argument("--output_root", type=str, default="analysis_results", help="Root directory for analysis outputs")
    parser.add_argument("--indices_path", type=str, nargs='+', default="None", help="Path to files containing indices for test split for each dataset")
    parser.add_argument("--mask_path", type=str, nargs='+', default="None", help="Path to files containing binary masks")
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Enable CUDA (GPU) if available")
    parser.add_argument("--group_cols", type=str, default="Sex,Site,Diagnosis", help="Comma-separated list of columns for group-wise analysis") # Added group_cols argument

    args = parser.parse_args()

    group_cols = [col.strip() for col in args.group_cols.split(',')] 

    analyzer = BrainAgeAnalyzer(
        validation_csv=args.validation_csv,
        validation_img_dir=args.validation_img_dir,
        model_dir=args.model_dir,
        output_root=args.output_root,
        use_cuda=args.use_cuda,
        group_columns=group_cols,
        indices_path=args.indices_path,
        mask_path=args.mask_path,
    )
    logging.info("Starting Brain Age Analysis...")
    analyzer.run_all_analyses()
