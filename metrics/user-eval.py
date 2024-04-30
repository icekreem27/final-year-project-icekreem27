import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# function to show a bar chart
def show_bar_chart(df):
    plt.style.use('seaborn-v0_8')

    pivot_table = df.pivot_table(index='question_number', columns='model', aggfunc='size', fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_table.plot(kind='bar', ax=ax, width=0.8)

    # Set difficulty levels
    difficulty = {i: 'Easy' for i in range(1, 4)}
    difficulty.update({i: 'Medium' for i in range(4, 7)})
    difficulty.update({i: 'Hard' for i in range(7, 11)})

    # Update x-axis labels to include difficulty levels
    ax.set_xticklabels([f'Q{idx} - {difficulty.get(idx, "Unknown")}' for idx in pivot_table.index], rotation=45, fontsize=12)

    ax.set_title('Model Picks per Question', fontsize=16)
    ax.set_xlabel('Question Number and Difficulty', fontsize=14)
    ax.set_ylabel('Number of Picks', fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Models", title_fontsize='13', fontsize='11', loc='upper right', edgecolor='black', frameon = True)
    
    # Set border properties
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')

    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.grid(False)
    
    y_values = [1, 2, 3, 4, 5, 6, 7 , 8, 9]
    for y in y_values:
        ax.axhline(y=y, color='gray', linewidth=0.4, linestyle='--')
    
    plt.savefig('Model_Picks_per_Question.png')
     
    plt.tight_layout()
    plt.show()

# function to show a pie chart
def show_pie_chart(df):
    plt.style.use('seaborn-v0_8')

    model_counts = df['model'].value_counts()

    fig, ax = plt.subplots()
    model_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, counterclock=False, shadow=False)
    ax.set_ylabel('')
    ax.set_title('Overall Model Picks Percentage', fontsize=16)
    
    plt.savefig('Overall_Model_Picks.png')

    plt.tight_layout()
    plt.show()

def main():
    model_renames = {
        'ft:gpt-3.5-turbo-0125:personal::90bpxw0O': 'RAG + FT',
        'gpt-3.5-turbo': 'RAG Only',
        'base': 'gpt-3.5-turbo'
    }
    df = pd.read_csv('Datasets/User Evaluation/evaluation_results.csv')

    df['model'] = df['model'].replace(model_renames)
    
    show_bar_chart(df)
    show_pie_chart(df)

if __name__ == '__main__':
    main()
