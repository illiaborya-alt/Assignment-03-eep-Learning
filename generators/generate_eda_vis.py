import os
import sys
import pandas as pd

# Налаштування шляхів, щоб Python бачив пакет src із кореня проєкту
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import matplotlib.pyplot as plt
from src.preprocessing import WordTokenizer
from src.dataset import load_dataset_csv

# Виправляємо шлях до data (вказуємо через parent_dir)
DATA_DIR = os.path.join(parent_dir, "data")
FIGURES_DIR = os.path.join(parent_dir, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
def generate_real_eda_figures():
    print("📊 Читаємо дані з папки data/ та генеруємо реальні EDA візуалізації...")

    # 1. Завантажуємо частину датасету для аналізу (наприклад, український датасет або тестовий)
    train_path = os.path.join(DATA_DIR, "uk_geo_dataset.csv")
    if os.path.exists(train_path):
        df = load_dataset_csv("uk_geo_dataset.csv", DATA_DIR, nrows=50000) # беремо вибірку для швидкості
    else:
        # Альтернатива, якщо є тільки тестовий або інший файл
        df = load_dataset_csv("test.csv", DATA_DIR)

    # Ініціалізуємо твій токенайзер
    tokenizer = WordTokenizer(lowercase=False)
    
    # Рахуємо реальну довжину послідовностей (кількість слів у кожному тексті)
    print("⏳ Обчислюємо довжини послідовностей...")
    seq_lengths = [len(tokenizer(text)) for text in df['text'].dropna().tolist()]

    # --- ГРАФІК 1: Реальний розподіл довжин послідовностей ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(seq_lengths, bins=40, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axvline(128, color='red', linestyle='--', linewidth=2, label='Selected max_len = 128')
    
    ax.set_title('Real Sequence Length Distribution (from data/corpus)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Words per Text', fontweight='bold')
    ax.set_ylabel('Frequency (Log Scale)', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eda_seq_lengths.png'), dpi=300)
    plt.close()

    # --- ГРАФІК 2: Реальний аналіз наявності локацій (або міток) ---
    if 'loc_markers' in df.columns:
        def has_location(markers):
            s = str(markers).strip()
            return 1 if (s and s != '[]' and s != 'nan') else 0
        
        df['has_loc'] = df['loc_markers'].apply(has_location)
        counts = df['has_loc'].value_counts()
        
        labels = ['No Location (Background / O)', 'Contains Location (Entity)']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['#95a5a6', '#2ecc71']

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
               textprops={'fontweight': 'bold'}, wedgeprops={'edgecolor': 'black'})
        ax.set_title('Corpus Balance: Texts with vs without Locations', fontsize=12, fontweight='bold', pad=10)

        fig.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'eda_class_distribution.png'), dpi=300)
        plt.close()

    print(f"✅ Реальнi EDA візуалізації успішно згенеровано та збережено у: {FIGURES_DIR}/")

if __name__ == "__main__":
    generate_real_eda_figures()