import sqlite3

def reset_database(db_name="database.db"):
    """
    Menghapus seluruh data history & results_detail
    TANPA menghapus struktur tabel
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    try:
        c.execute("DELETE FROM results_detail")
        c.execute("DELETE FROM history")

        c.execute("DELETE FROM sqlite_sequence WHERE name='results_detail'")
        c.execute("DELETE FROM sqlite_sequence WHERE name='history'")

        conn.commit()
        print("✅ Database berhasil di-reset (data dihapus)")
    except Exception as e:
        conn.rollback()
        print("❌ Gagal reset database:", e)
    finally:
        conn.close()


reset_database('analysis_history.db')