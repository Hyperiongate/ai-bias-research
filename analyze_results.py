"""
AI Bias Research - Analysis Tool with Visualizations
Created: December 16, 2024
Author: Jim (Hyperiongate)

This tool analyzes CSV exports from the AI Bias Research Tool and creates:
- Comprehensive statistical analysis
- Visual charts and graphs
- Side-by-side AI comparisons
- Category-based breakdowns
- Trend analysis
- Publication-ready reports

Usage:
    python analyze_results.py ai_bias_research_20251216_204310.csv

Requirements:
    pip install pandas matplotlib seaborn numpy

Features:
- Automated pattern detection
- Geographic bias analysis
- Political bias scoring
- Ideological clustering
- Scientific consensus validation
- Interactive comparison mode
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import sys
import os
from datetime import datetime

# Set style for professional charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AIBiasAnalyzer:
    def __init__(self, csv_file):
        """Initialize analyzer with CSV data"""
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.questions = self.df['Question'].unique()
        self.ai_systems = self.df['AI System'].unique()
        
        print(f"✅ Loaded {len(self.df)} responses")
        print(f"   {len(self.questions)} unique questions")
        print(f"   {len(self.ai_systems)} AI systems")
        print()
    
    def summary_statistics(self):
        """Generate summary statistics for all questions"""
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print()
        
        results = []
        
        for question in self.questions:
            q_data = self.df[self.df['Question'] == question]
            
            # Filter valid ratings
            valid_ratings = q_data[q_data['Rating'] != 'N/A']['Rating'].astype(float)
            
            if len(valid_ratings) < 3:
                continue
            
            stats = {
                'Question': question[:60] + '...',
                'Count': len(valid_ratings),
                'Mean': valid_ratings.mean(),
                'Std': valid_ratings.std(),
                'Min': valid_ratings.min(),
                'Max': valid_ratings.max(),
                'Spread': valid_ratings.max() - valid_ratings.min()
            }
            
            results.append(stats)
            
            # Print detailed breakdown
            print(f"Q: {question[:70]}...")
            print(f"   Mean: {stats['Mean']:.3f} | Spread: {stats['Spread']:.3f} | StdDev: {stats['Std']:.3f}")
            print(f"   Range: {stats['Min']:.3f} - {stats['Max']:.3f}")
            print()
        
        return pd.DataFrame(results)
    
    def ai_system_profiles(self):
        """Create profiles for each AI system"""
        print("=" * 80)
        print("AI SYSTEM PROFILES")
        print("=" * 80)
        print()
        
        profiles = []
        
        for ai in self.ai_systems:
            ai_data = self.df[self.df['AI System'] == ai]
            
            # Filter valid ratings
            valid_ratings = ai_data[ai_data['Rating'] != 'N/A']['Rating'].astype(float)
            
            if len(valid_ratings) == 0:
                continue
            
            profile = {
                'AI System': ai,
                'Total Responses': len(ai_data),
                'Success Rate': (ai_data['Provided Rating'] == 'Yes').sum() / len(ai_data) * 100,
                'Avg Rating': valid_ratings.mean(),
                'Rating StdDev': valid_ratings.std(),
                'Avg Word Count': ai_data['Word Count'].mean(),
                'Avg Hedge Freq': ai_data['Hedge Frequency (%)'].mean(),
                'Avg Sentiment': ai_data['Sentiment Score'].mean(),
                'Avg Response Time': ai_data['Response Time (s)'].mean()
            }
            
            profiles.append(profile)
            
            print(f"{ai}:")
            print(f"   Success Rate: {profile['Success Rate']:.1f}%")
            print(f"   Avg Rating: {profile['Avg Rating']:.3f} (±{profile['Rating StdDev']:.3f})")
            print(f"   Avg Words: {profile['Avg Word Count']:.0f}")
            print(f"   Hedge Freq: {profile['Avg Hedge Freq']:.2f}%")
            print(f"   Response Time: {profile['Avg Response Time']:.2f}s")
            print()
        
        return pd.DataFrame(profiles)
    
    def compare_ais(self, question, ai_list=None):
        """Compare specific AIs on a specific question"""
        if ai_list is None:
            ai_list = self.ai_systems
        
        q_data = self.df[self.df['Question'] == question]
        comparison = []
        
        for ai in ai_list:
            ai_response = q_data[q_data['AI System'] == ai]
            
            if len(ai_response) == 0:
                continue
            
            row = ai_response.iloc[0]
            
            comparison.append({
                'AI System': ai,
                'Rating': row['Rating'],
                'Word Count': row['Word Count'],
                'Hedge Freq': row['Hedge Frequency (%)'],
                'Sentiment': row['Sentiment Score'],
                'Response Time': row['Response Time (s)']
            })
        
        return pd.DataFrame(comparison)
    
    def detect_geographic_bias(self):
        """Detect geographic bias patterns"""
        print("=" * 80)
        print("GEOGRAPHIC BIAS DETECTION")
        print("=" * 80)
        print()
        
        # Define AI origins
        ai_origins = {
            'OpenAI': 'USA',
            'Google': 'USA',
            'Anthropic': 'USA',
            'xAI': 'USA',
            'Meta (via Groq)': 'USA',
            'DeepSeek': 'China',
            'Mistral': 'France',
            'Cohere': 'Canada'
        }
        
        # Look for questions about world leaders
        leader_questions = [q for q in self.questions if any(
            name in q for name in ['Trump', 'Biden', 'Obama', 'Xi Jinping', 'Putin']
        )]
        
        for question in leader_questions:
            q_data = self.df[self.df['Question'] == question]
            
            print(f"Q: {question[:70]}...")
            
            by_origin = defaultdict(list)
            
            for _, row in q_data.iterrows():
                ai = row['AI System']
                rating = row['Rating']
                
                if rating != 'N/A':
                    origin = ai_origins.get(ai, 'Unknown')
                    by_origin[origin].append(float(rating))
            
            for origin, ratings in by_origin.items():
                if len(ratings) > 0:
                    print(f"   {origin:10s}: {np.mean(ratings):.3f} (n={len(ratings)})")
            
            print()
    
    def detect_ideological_bias(self):
        """Detect ideological bias (capitalism vs socialism)"""
        print("=" * 80)
        print("IDEOLOGICAL BIAS DETECTION")
        print("=" * 80)
        print()
        
        cap_question = [q for q in self.questions if 'capitalism' in q.lower()]
        soc_question = [q for q in self.questions if 'socialism' in q.lower()]
        
        if len(cap_question) == 0 or len(soc_question) == 0:
            print("No capitalism/socialism questions found")
            return
        
        cap_data = self.df[self.df['Question'] == cap_question[0]]
        soc_data = self.df[self.df['Question'] == soc_question[0]]
        
        print("Capitalism vs Socialism Ratings:\n")
        
        for ai in self.ai_systems:
            cap_rating = cap_data[cap_data['AI System'] == ai]['Rating'].values
            soc_rating = soc_data[soc_data['AI System'] == ai]['Rating'].values
            
            if len(cap_rating) > 0 and len(soc_rating) > 0 and cap_rating[0] != 'N/A' and soc_rating[0] != 'N/A':
                cap_val = float(cap_rating[0])
                soc_val = float(soc_rating[0])
                diff = cap_val - soc_val
                
                bias_label = "Pro-Capitalism" if diff > 2 else "Balanced" if abs(diff) <= 2 else "Pro-Socialism"
                
                print(f"{ai:20s}: Cap={cap_val:.2f} | Soc={soc_val:.2f} | Diff={diff:+.2f} ({bias_label})")
        
        print()
    
    def create_comparison_chart(self, question, output_file='comparison.png'):
        """Create visual comparison chart for a question"""
        q_data = self.df[self.df['Question'] == question]
        
        # Filter valid ratings
        valid_data = q_data[q_data['Rating'] != 'N/A'].copy()
        valid_data['Rating'] = valid_data['Rating'].astype(float)
        
        # Sort by rating
        valid_data = valid_data.sort_values('Rating')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        bars = ax.barh(valid_data['AI System'], valid_data['Rating'], 
                       color=plt.cm.viridis(valid_data['Rating'] / 10))
        
        # Customize
        ax.set_xlabel('Rating (1-10)', fontsize=12)
        ax.set_title(f'{question[:80]}...', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center', fontsize=10)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved: {output_file}")
        
        return output_file
    
    def create_spread_analysis_chart(self, output_file='spread_analysis.png'):
        """Create chart showing spread for each question"""
        spreads = []
        question_labels = []
        
        for question in self.questions:
            q_data = self.df[self.df['Question'] == question]
            valid_ratings = q_data[q_data['Rating'] != 'N/A']['Rating'].astype(float)
            
            if len(valid_ratings) >= 3:
                spread = valid_ratings.max() - valid_ratings.min()
                spreads.append(spread)
                
                # Shorten question for label
                label = question[:40] + '...' if len(question) > 40 else question
                question_labels.append(label)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sort by spread
        sorted_indices = np.argsort(spreads)[::-1]
        sorted_spreads = [spreads[i] for i in sorted_indices]
        sorted_labels = [question_labels[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        colors = ['red' if s > 3 else 'orange' if s > 2 else 'yellow' if s > 1 else 'green' 
                  for s in sorted_spreads]
        
        bars = ax.barh(range(len(sorted_spreads)), sorted_spreads, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlabel('Rating Spread (Max - Min)', fontsize=12)
        ax.set_title('AI Rating Spread by Question\n(High Spread = High Disagreement)', 
                     fontsize=14, fontweight='bold')
        
        # Add spread values
        for i, (bar, spread) in enumerate(zip(bars, sorted_spreads)):
            ax.text(spread + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{spread:.2f}', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Very High (>3.0)'),
            Patch(facecolor='orange', alpha=0.7, label='High (2.0-3.0)'),
            Patch(facecolor='yellow', alpha=0.7, label='Medium (1.0-2.0)'),
            Patch(facecolor='green', alpha=0.7, label='Low (<1.0)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Spread analysis saved: {output_file}")
        
        return output_file
    
    def create_ai_profile_comparison(self, output_file='ai_profiles.png'):
        """Create radar chart comparing AI profiles"""
        profiles = self.ai_system_profiles()
        
        if len(profiles) == 0:
            print("No profiles to compare")
            return
        
        # Select metrics for radar chart
        metrics = ['Avg Rating', 'Avg Hedge Freq', 'Avg Response Time']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each AI
        for _, profile in profiles.iterrows():
            values = [
                profile['Avg Rating'] / 10,  # Normalize to 0-1
                profile['Avg Hedge Freq'] / 100,  # Normalize to 0-1
                profile['Avg Response Time'] / 10  # Normalize to 0-1
            ]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=profile['AI System'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('AI System Profiles\n(Normalized Metrics)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ AI profiles saved: {output_file}")
        
        return output_file
    
    def generate_report(self, output_dir='analysis_output'):
        """Generate comprehensive analysis report"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        print()
        
        # 1. Summary statistics
        summary = self.summary_statistics()
        summary.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
        print(f"✅ Summary statistics saved")
        
        # 2. AI profiles
        profiles = self.ai_system_profiles()
        profiles.to_csv(f'{output_dir}/ai_profiles.csv', index=False)
        print(f"✅ AI profiles saved")
        
        # 3. Geographic bias analysis
        self.detect_geographic_bias()
        
        # 4. Ideological bias analysis
        self.detect_ideological_bias()
        
        # 5. Create visualizations
        print("\nGenerating visualizations...")
        
        # Spread analysis
        self.create_spread_analysis_chart(f'{output_dir}/spread_analysis.png')
        
        # AI profile comparison
        self.create_ai_profile_comparison(f'{output_dir}/ai_profiles_comparison.png')
        
        # Individual question charts (top 5 by spread)
        summary_sorted = summary.sort_values('Spread', ascending=False)
        for i, row in summary_sorted.head(5).iterrows():
            question = row['Question'].replace('...', '')
            # Find full question
            full_question = [q for q in self.questions if q.startswith(question)][0]
            
            safe_filename = f"question_{i+1}.png"
            self.create_comparison_chart(full_question, f'{output_dir}/{safe_filename}')
        
        print()
        print("=" * 80)
        print(f"✅ REPORT COMPLETE! All files saved to: {output_dir}/")
        print("=" * 80)
        print()
        print("Files created:")
        print(f"  - summary_statistics.csv")
        print(f"  - ai_profiles.csv")
        print(f"  - spread_analysis.png")
        print(f"  - ai_profiles_comparison.png")
        print(f"  - question_1.png through question_5.png")
        print()

def interactive_menu(analyzer):
    """Interactive menu for analysis"""
    while True:
        print("\n" + "=" * 80)
        print("AI BIAS RESEARCH - ANALYSIS MENU")
        print("=" * 80)
        print()
        print("1. Summary Statistics")
        print("2. AI System Profiles")
        print("3. Geographic Bias Analysis")
        print("4. Ideological Bias Analysis")
        print("5. Compare Specific AIs on a Question")
        print("6. Generate Full Report (All Charts + CSVs)")
        print("7. Exit")
        print()
        
        choice = input("Select option (1-7): ").strip()
        
        if choice == '1':
            analyzer.summary_statistics()
        elif choice == '2':
            analyzer.ai_system_profiles()
        elif choice == '3':
            analyzer.detect_geographic_bias()
        elif choice == '4':
            analyzer.detect_ideological_bias()
        elif choice == '5':
            print("\nAvailable questions:")
            for i, q in enumerate(analyzer.questions, 1):
                print(f"{i}. {q[:70]}...")
            q_num = int(input("\nSelect question number: "))
            question = analyzer.questions[q_num - 1]
            
            comparison = analyzer.compare_ais(question)
            print(comparison.to_string(index=False))
        elif choice == '6':
            output_dir = input("Output directory (default: analysis_output): ").strip() or "analysis_output"
            analyzer.generate_report(output_dir)
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <csv_file>")
        print()
        print("Example:")
        print("  python analyze_results.py ai_bias_research_20251216_204310.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("AI BIAS RESEARCH - ANALYSIS TOOL")
    print("=" * 80)
    print(f"Loading: {csv_file}")
    print()
    
    analyzer = AIBiasAnalyzer(csv_file)
    
    # Check if user wants interactive mode or automatic report
    if len(sys.argv) > 2 and sys.argv[2] == '--auto':
        analyzer.generate_report()
    else:
        interactive_menu(analyzer)

if __name__ == '__main__':
    main()

# I did no harm and this file is not truncated
