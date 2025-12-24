import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import argparse
import os

def load_and_preprocess_data(filepath):
    """
    Memuat dan memproses data CSV menjadi format yang siap dianalisis.
    
    Args:
        filepath (str): Path ke file CSV
        
    Returns:
        tuple: (pivot_df, rides, state_vectors)
    """
    print("=" * 60)
    print("LANGKAH 1: MEMUAT DAN MEMPROSES DATA")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✓ Data dimuat: {len(df)} baris, {df['timestamp'].nunique()} timestamp unik")
    
    # Pivot data untuk mendapatkan vektor wait time per timestamp
    pivot_df = df.pivot(index='timestamp', columns='ride_name', values='wait_time').fillna(0)
    rides = pivot_df.columns.tolist()
    
    print(f"✓ Wahana yang dianalisis ({len(rides)}):")
    for i, ride in enumerate(rides, 1):
        print(f"  {i}. {ride}")
    
    # Normalisasi data menjadi probabilitas (Vektor Keadaan)
    # v_t = wait_time_i / sum(wait_times)
    # Hindari pembagian dengan nol
    row_sums = pivot_df.sum(axis=1).replace(0, 1)  # Ganti 0 dengan 1 untuk menghindari div by zero
    state_vectors = pivot_df.div(row_sums, axis=0).values
    
    print(f"✓ Vektor keadaan dibentuk: {state_vectors.shape[0]} observasi")
    print(f"✓ Rentang waktu: {df['timestamp'].min()} hingga {df['timestamp'].max()}")
    print()
    
    return pivot_df, rides, state_vectors

def estimate_transition_matrix_v1(state_vectors, rides):
    """
    Metode 1: Estimasi matriks transisi berdasarkan distribusi rata-rata.
    
    Asumsi: Probabilitas transisi ke wahana i proporsional dengan 
    daya tarik relatif wahana tersebut (diukur dari rata-rata waktu tunggu).
    
    Args:
        state_vectors (np.ndarray): Array vektor keadaan
        rides (list): Daftar nama wahana
        
    Returns:
        np.ndarray: Matriks transisi A (5x5)
    """
    print("=" * 60)
    print("LANGKAH 2: ESTIMASI MATRIKS TRANSISI (Metode: Distribusi Rata-rata)")
    print("=" * 60)
    
    # Hitung distribusi probabilitas rata-rata
    avg_distribution = np.mean(state_vectors, axis=0)
    
    print("Distribusi probabilitas rata-rata:")
    for ride, prob in zip(rides, avg_distribution):
        print(f"  - {ride}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Bentuk matriks transisi: setiap kolom adalah avg_distribution
    # Interpretasi: dari wahana manapun, probabilitas menuju wahana i
    # proporsional dengan daya tarik wahana i
    A = np.tile(avg_distribution.reshape(-1, 1), (1, len(rides)))
    
    # Pastikan matriks stokastik kolom (sum setiap kolom = 1)
    A = A / A.sum(axis=0)
    
    print("\nMatriks Transisi A (5×5):")
    print("(Baris: tujuan, Kolom: asal)")
    print(pd.DataFrame(A, index=rides, columns=rides).round(4))
    print()
    
    return A

def estimate_transition_matrix_v2(state_vectors, rides):
    """
    Metode 2: Estimasi matriks transisi dari perubahan state antar waktu.
    
    Menghitung rata-rata transisi empiris: v_{t+1} ≈ A * v_t
    Solusi least squares: A = Σ(v_{t+1} * v_t^T) * [Σ(v_t * v_t^T)]^{-1}
    
    Args:
        state_vectors (np.ndarray): Array vektor keadaan
        rides (list): Daftar nama wahana
        
    Returns:
        np.ndarray: Matriks transisi A (5x5)
    """
    print("=" * 60)
    print("LANGKAH 2: ESTIMASI MATRIKS TRANSISI (Metode: Least Squares)")
    print("=" * 60)
    
    n = len(state_vectors)
    if n < 2:
        print("ERROR: Tidak cukup data untuk estimasi transisi temporal.")
        return None
    
    # Pisahkan data menjadi v_t (current) dan v_{t+1} (next)
    V_current = state_vectors[:-1].T  # (5, n-1)
    V_next = state_vectors[1:].T      # (5, n-1)
    
    # Estimasi A menggunakan least squares
    # V_next ≈ A * V_current
    # A = V_next * V_current^T * (V_current * V_current^T)^{-1}
    
    try:
        VVT = V_current @ V_current.T
        # Tambahkan regularisasi kecil untuk stabilitas numerik
        VVT_inv = np.linalg.inv(VVT + 1e-8 * np.eye(len(rides)))
        A = V_next @ V_current.T @ VVT_inv
        
        # Normalisasi untuk memastikan matriks stokastik kolom
        A = np.maximum(A, 0)  # Paksa non-negatif
        col_sums = A.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Hindari pembagian nol
        A = A / col_sums
        
        print("Matriks Transisi A (5×5):")
        print("(Baris: tujuan, Kolom: asal)")
        print(pd.DataFrame(A, index=rides, columns=rides).round(4))
        print()
        
        return A
        
    except np.linalg.LinAlgError:
        print("ERROR: Matriks singular, tidak dapat menghitung inverse.")
        print("Menggunakan metode alternatif...")
        return estimate_transition_matrix_v1(state_vectors, rides)

def analyze_eigenvalues(A, rides):
    """
    Melakukan analisis spektral pada matriks transisi.
    
    Args:
        A (np.ndarray): Matriks transisi
        rides (list): Daftar nama wahana
        
    Returns:
        tuple: (eigenvalues, eigenvectors, stationary_vector)
    """
    print("=" * 60)
    print("LANGKAH 3: ANALISIS SPEKTRAL (Nilai Eigen dan Vektor Eigen)")
    print("=" * 60)
    
    # Hitung nilai eigen dan vektor eigen
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Spektrum Nilai Eigen:")
    for i, ev in enumerate(eigenvalues, 1):
        if np.isreal(ev):
            print(f"  λ_{i} = {np.real(ev):.6f}")
        else:
            print(f"  λ_{i} = {ev:.6f}")
    
    # Cari nilai eigen dominan (yang paling mendekati 1)
    idx_dominant = np.argmin(np.abs(eigenvalues - 1.0))
    lambda_dominant = eigenvalues[idx_dominant]
    
    print(f"\nNilai eigen dominan: λ_1 = {np.real(lambda_dominant):.6f}")
    
    if np.abs(lambda_dominant - 1.0) > 0.01:
        print("PERINGATAN: Nilai eigen dominan tidak cukup dekat dengan 1.")
        print("Matriks mungkin bukan matriks stokastik yang valid.")
    
    # Ekstrak vektor eigen dominan (vektor stasioner)
    stationary_vector = np.real(eigenvectors[:, idx_dominant])
    
    # Normalisasi agar jumlah komponen = 1 (distribusi probabilitas)
    stationary_vector = np.abs(stationary_vector)  # Pastikan positif
    stationary_vector = stationary_vector / stationary_vector.sum()
    
    print("\nVektor Eigen Stasioner (Distribusi Titik Jenuh):")
    for ride, prob in zip(rides, stationary_vector):
        print(f"  - {ride}: {prob:.6f} ({prob*100:.2f}%)")
    
    # Identifikasi titik jenuh (wahana dengan probabilitas tertinggi)
    max_idx = np.argmax(stationary_vector)
    print(f"\n🎯 TITIK JENUH: {rides[max_idx]} ({stationary_vector[max_idx]*100:.2f}%)")
    print()
    
    return eigenvalues, eigenvectors, stationary_vector

def verify_convergence(A, state_vectors, stationary_vector, rides, max_iterations=100):
    """
    Memverifikasi konvergensi dengan simulasi perkalian matriks.
    
    Args:
        A (np.ndarray): Matriks transisi
        state_vectors (np.ndarray): Vektor keadaan awal
        stationary_vector (np.ndarray): Vektor stasioner teoritis
        rides (list): Daftar nama wahana
        max_iterations (int): Jumlah maksimum iterasi
    """
    print("=" * 60)
    print("LANGKAH 4: VERIFIKASI KONVERGENSI")
    print("=" * 60)
    
    # Gunakan kondisi awal dari observasi pertama
    v0 = state_vectors[0].copy()
    
    print(f"Kondisi awal (v_0):")
    for ride, prob in zip(rides, v0):
        print(f"  - {ride}: {prob:.6f} ({prob*100:.2f}%)")
    
    print(f"\nMelakukan iterasi v_{{n+1}} = A * v_n hingga konvergen...")
    
    v_current = v0.copy()
    converged = False
    
    for iteration in range(1, max_iterations + 1):
        v_next = A @ v_current
        
        # Cek konvergensi (norma perbedaan < threshold)
        diff = np.linalg.norm(v_next - v_current)
        
        if iteration % 10 == 0 or diff < 1e-6:
            print(f"  Iterasi {iteration}: ||v_{iteration} - v_{iteration-1}|| = {diff:.8f}")
        
        if diff < 1e-6:
            print(f"\n✓ Konvergen pada iterasi ke-{iteration}")
            converged = True
            v_current = v_next
            break
        
        v_current = v_next
    
    if not converged:
        print(f"\n⚠ Tidak konvergen setelah {max_iterations} iterasi")
    
    print(f"\nVektor hasil simulasi (v_∞):")
    for ride, prob in zip(rides, v_current):
        print(f"  - {ride}: {prob:.6f} ({prob*100:.2f}%)")
    
    print(f"\nPerbandingan dengan vektor stasioner teoritis:")
    error = np.linalg.norm(v_current - stationary_vector)
    print(f"  Error (norma L2): {error:.8f}")
    
    if error < 0.01:
        print("  ✓ Sangat baik! Simulasi sesuai dengan teori.")
    elif error < 0.05:
        print("  ✓ Baik. Perbedaan minor dapat diterima.")
    else:
        print("  ⚠ Perbedaan signifikan. Periksa kembali matriks transisi.")
    print()

def visualize_results(state_vectors, stationary_vector, rides):
    """
    Membuat visualisasi perbandingan kondisi awal vs kondisi jenuh.
    
    Args:
        state_vectors (np.ndarray): Array vektor keadaan
        stationary_vector (np.ndarray): Vektor stasioner
        rides (list): Daftar nama wahana
    """
    print("=" * 60)
    print("LANGKAH 5: VISUALISASI HASIL")
    print("=" * 60)
    
    # Hitung distribusi rata-rata dari observasi
    avg_observed = np.mean(state_vectors, axis=0)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(rides))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, avg_observed * 100, width, 
                   label='Distribusi Observasi Rata-rata', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, stationary_vector * 100, width,
                   label='Distribusi Stasioner (Titik Jenuh)', alpha=0.8, color='coral')
    
    # Customization
    ax.set_xlabel('Wahana', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probabilitas (%)', fontsize=11, fontweight='bold')
    ax.set_title('Perbandingan Distribusi Pengunjung: Observasi vs Prediksi Steady-State', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([r.split('™')[0].split('®')[0].strip() for r in rides], 
                        rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Tambahkan nilai di atas bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    # Ensure outputs directory exists inside the workspace
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'visualisasi_distribusi.png')
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grafik tersimpan: {save_path}")
    except Exception as e:
        print(f"ERROR: Gagal menyimpan grafik: {e}")
    finally:
        plt.close(fig)
    print()

def main():
    """Fungsi utama untuk menjalankan seluruh analisis."""
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  ANALISIS ALGEBRA LINIER: DISTRIBUSI PENGUNJUNG WAHANA  ".center(58) + "║")
    print("║" + "          Metode Matriks Transisi & Analisis Spektral          ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Parse optional filepath argument; otherwise try to locate the file
    parser = argparse.ArgumentParser(description='Analisis distribusi pengunjung wahana')
    parser.add_argument('filepath', nargs='?', help='Path ke file CSV data (optional)')
    args = parser.parse_args()

    default_name = 'data_antrean_5_wahana.csv'
    def find_data_file(provided_path=None, name=default_name, max_up=4):
        # If user provided explicit path, try it first
        if provided_path:
            if os.path.isabs(provided_path):
                if os.path.exists(provided_path):
                    return provided_path
            else:
                cand = os.path.abspath(provided_path)
                if os.path.exists(cand):
                    return cand

        # Check current working directory
        cwd_candidate = os.path.join(os.getcwd(), name)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        # Check script directory and parents up to max_up levels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cur = script_dir
        for _ in range(max_up + 1):
            cand = os.path.join(cur, name)
            if os.path.exists(cand):
                return cand
            cur = os.path.dirname(cur)

        return None

    filepath = find_data_file(args.filepath)
    
    try:
        # LANGKAH 1: Load dan preprocess data
        if filepath is None:
            raise FileNotFoundError(default_name)
        pivot_df, rides, state_vectors = load_and_preprocess_data(filepath)
        
        # LANGKAH 2: Estimasi matriks transisi
        # Pilih salah satu metode (v2 lebih sophisticated)
        print("Pilih metode estimasi matriks transisi:")
        print("  [1] Distribusi rata-rata (sederhana)")
        print("  [2] Least squares dari transisi temporal (kompleks)")
        
        # Untuk otomatis, gunakan metode 2
        A = estimate_transition_matrix_v2(state_vectors, rides)
        
        if A is None:
            print("Gagal mengestimasi matriks. Program dihentikan.")
            return
        
        # Validasi matriks stokastik
        col_sums = A.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            print("PERINGATAN: Matriks bukan stokastik kolom yang sempurna!")
            print(f"Sum kolom: {col_sums}")
        
        # LANGKAH 3: Analisis spektral
        eigenvalues, eigenvectors, stationary_vector = analyze_eigenvalues(A, rides)
        
        # LANGKAH 4: Verifikasi konvergensi
        verify_convergence(A, state_vectors, stationary_vector, rides)
        
        # LANGKAH 5: Visualisasi
        visualize_results(state_vectors, stationary_vector, rides)
        
        # RINGKASAN AKHIR
        print("=" * 60)
        print("RINGKASAN HASIL")
        print("=" * 60)
        print(f"✓ Jumlah observasi: {len(state_vectors)}")
        print(f"✓ Jumlah wahana: {len(rides)}")
        print(f"✓ Nilai eigen dominan: {np.real(eigenvalues[np.argmin(np.abs(eigenvalues - 1.0))]):.6f}")
        print(f"\n📊 Distribusi Stasioner (Titik Jenuh):")
        
        sorted_indices = np.argsort(stationary_vector)[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"  {rank}. {rides[idx]}: {stationary_vector[idx]*100:.2f}%")
        
        print("\n" + "=" * 60)
        print("ANALISIS SELESAI")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"ERROR: File '{filepath}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan skrip pengambilan data terlebih dahulu.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()