import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud, STOPWORDS

# Simpan data user dalam dictionary
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': 'admin'}  # Inisialisasi dengan user admin

# Fungsi untuk memuat data
def load_data(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path, delimiter=';')
        if 'stemming' in dataset.columns and 'Sentimen' in dataset.columns:
            dataset = dataset.dropna(subset=['stemming', 'Sentimen'])
            return dataset
        else:
            st.error("Kolom 'stemming' dan/atau 'Sentimen' tidak ditemukan dalam dataset.")
            return None
    except FileNotFoundError:
        st.error(f"File '{dataset_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
        return None

# Fungsi untuk melatih model
def train_model():
    dataset_path = 'Hasil-Labeling.csv'
    dataset = load_data(dataset_path)
    if dataset is not None:
        text_data = dataset['stemming']
        labels = dataset['Sentimen']

        X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)

        vectorizer = CountVectorizer(ngram_range=(1, 1))
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        st.session_state['vectorizer'] = vectorizer
        st.session_state['model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

# Fungsi untuk menampilkan Word Cloud
def create_wordcloud(text, title):
    stopwords = set(STOPWORDS)
    wc = WordCloud(stopwords=stopwords, background_color="black", max_words=500, width=800, height=500)
    wc.generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    st.pyplot(fig)

# Fungsi untuk menampilkan Word Cloud berdasarkan sentimen
def display_wordcloud_sentiment(sentiment):
    dataset = st.session_state.get('dataset')
    if dataset is not None:
        data = dataset[dataset['Sentimen'] == sentiment]['stemming'].tolist()
        text = ' '.join(data)
        create_wordcloud(text, f'Word Cloud untuk Sentimen: {sentiment}')

# Fungsi untuk halaman utama
def main_page(is_admin):
    st.sidebar.title("Menu")
    if is_admin:
        option = st.sidebar.radio('Pilih Halaman', ('DataFrame', 'Confusion Matrix', 'Persentase Diagram', 'Word Cloud', 'Logout'))
    else:
        option = st.sidebar.radio('Pilih Halaman', ('DataFrame', 'Logout'))
    
    dataset_path = 'Hasil-Labeling.csv'
    dataset = load_data(dataset_path)

    if option == 'Confusion Matrix':
        st.markdown("<h1 style='text-align: center;'>Confusion Matrix Data Testing Dan Data Training</h1>", unsafe_allow_html=True)
        st.sidebar.success("Kamu sedang berada pada menu Confusion Matrix")

        if dataset is not None:
            text_data = dataset['stemming']
            labels = dataset['Sentimen']

            X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)

            vectorizer = CountVectorizer(ngram_range=(1, 1))  # unigram
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            model = MultinomialNB()
            model.fit(X_train_vectorized, y_train)

            predictions = model.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'Accuracy Testing Data: {accuracy:.2f}')
            st.write('Classification Report Testing Data:\n', classification_report(y_test, predictions))

            conf_matrix = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positive'], yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
            ax.set_title('Confusion Matrix Testing Data')
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            st.pyplot(fig)

            predictions_train = model.predict(X_train_vectorized)
            accuracy_train = accuracy_score(y_train, predictions_train)
            st.write(f'Accuracy Training Data: {accuracy_train:.2f}')
            st.write('Classification Report Training Data:\n', classification_report(y_train, predictions_train))

            conf_matrix_train = confusion_matrix(y_train, predictions_train)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
            ax.set_title('Confusion Matrix Training Data')
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            st.pyplot(fig)

    elif option == 'Persentase Diagram':
        st.sidebar.success("Kamu sedang berada di menu Persentase Diagram Masing-Masing Sentimen")
        st.markdown("<h1 style='text-align: center;'>Persentase Masing-Masing Sentimen</h1>", unsafe_allow_html=True)
        if dataset is not None:
            diagram = dataset['Sentimen'].value_counts().rename_axis('nilai_sentimen').reset_index(name='jumlah')
            label = diagram.nilai_sentimen
            nilai = diagram.jumlah

            fig, ax = plt.subplots()
            ax.pie(nilai, labels=label, autopct='%1.2f%%')
            st.pyplot(fig)

            sentiment_count = dataset['Sentimen'].value_counts()
            sns.set_style('whitegrid')

            fig, ax = plt.subplots(figsize=(6, 4))
            ax = sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='pastel')
            plt.title('Jumlah Analisis Sentimen', fontsize=14, pad=20)
            plt.xlabel('Class Sentimen', fontsize=12)
            plt.ylabel('Jumlah Tweet', fontsize=12)

            for i, count in enumerate(sentiment_count.values):
                ax.text(i, count + 0.10, str(count), ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.error("Data tidak tersedia. Mohon muat data terlebih dahulu.")

    elif option == 'Word Cloud':
        st.sidebar.success("Kamu sedang berada di menu Word Cloud")
        st.markdown("<h1 style='text-align: center;'>Word Cloud</h1>", unsafe_allow_html=True)
        if dataset is not None:
            Positif_text = ' '.join(dataset[dataset['Sentimen'] == 'Positif']['stemming'].tolist())
            Negatif_text = ' '.join(dataset[dataset['Sentimen'] == 'Negatif']['stemming'].tolist())
            Netral_text = ' '.join(dataset[dataset['Sentimen'] == 'Netral']['stemming'].tolist())

            create_wordcloud(Positif_text, 'Word Cloud Sentimen Positif')
            create_wordcloud(Negatif_text, 'Word Cloud Sentimen Negatif')
            create_wordcloud(Netral_text, 'Word Cloud Sentimen Netral')
        else:
            st.error("Data tidak tersedia. Mohon muat data terlebih dahulu.")

    elif option == 'DataFrame':
        st.markdown("<h1 style='text-align: center;'>DataFrame</h1>", unsafe_allow_html=True)
        dataset_path = 'Hasil-Labeling.csv'
        dataset = load_data(dataset_path)
        if dataset is not None:
            st.session_state['dataset'] = dataset
            
            if is_admin:
                st.write(dataset)  # Admin dapat melihat seluruh kolom
            else:
                st.write(dataset[['stemming']])  # Pengguna biasa hanya dapat melihat kolom 'stemming'

            if 'vectorizer' not in st.session_state or 'model' not in st.session_state:
                st.warning("Model belum dilatih. Melatih model sekarang...")
                train_model()

            if is_admin:  # Hanya pengguna admin yang melihat bagian ini
                st.markdown("<h2>Edit Data</h2>", unsafe_allow_html=True)
                row_number = st.number_input("Masukkan Nomor Baris yang Ingin Diedit", min_value=0, max_value=len(dataset)-1, step=1)
                selected_row = dataset.iloc[row_number]
                st.write("Data yang dipilih:", selected_row)
                
                new_sentiment = st.selectbox("Pilih Sentimen Baru", options=['Positif', 'Netral', 'Negatif'], index=['Positif', 'Netral', 'Negatif'].index(selected_row['Sentimen']))
                if st.button("Simpan Perubahan"):
                    dataset.at[row_number, 'Sentimen'] = new_sentiment
                    st.session_state['dataset'] = dataset  # Update session state dataset
                    dataset.to_csv('Hasil-Labeling.csv', index=False, sep=';')  # Simpan ke file
                    st.success("Data berhasil diubah!")
                    st.experimental_rerun()  # Rerun untuk merender ulang data frame
            
            else:  # Hanya pengguna non-admin yang melihat bagian ini
                user_input = st.text_input("Masukkan Tweet:", key="user_input")
                data_choice = st.radio("Pilih jenis data:", ('Testing', 'Training'))

                if st.button("Hasil"):
                    if user_input:
                        user_input_vector = st.session_state['vectorizer'].transform([user_input])
                        prediction = st.session_state['model'].predict(user_input_vector)
                        st.session_state['user_prediction'] = prediction[0]
                        st.success(f"Prediksi Sentimen: {prediction[0]}")

                        # Tampilkan akurasi model berdasarkan pilihan data
                        if data_choice == 'Testing':
                            X_test_vectorized = st.session_state['vectorizer'].transform(st.session_state['X_test'])
                            y_test = st.session_state['y_test']
                            predictions = st.session_state['model'].predict(X_test_vectorized)
                            accuracy = accuracy_score(y_test, predictions)
                            st.write(f'Accuracy Testing Data: {accuracy:.2f}')
                        elif data_choice == 'Training':
                            X_train_vectorized = st.session_state['vectorizer'].transform(st.session_state['X_train'])
                            y_train = st.session_state['y_train']
                            predictions_train = st.session_state['model'].predict(X_train_vectorized)
                            accuracy_train = accuracy_score(y_train, predictions_train)
                            st.write(f'Accuracy Training Data: {accuracy_train:.2f}')

                        # Tampilkan Word Cloud berdasarkan hasil prediksi sentimen
                        display_wordcloud_sentiment(st.session_state['user_prediction'])
                    else:
                        st.warning("Mohon masukkan teks tweet.")

    elif option == 'Logout':
        st.session_state['logged_in'] = False
        st.success("Anda telah logout.")
        st.experimental_rerun()

# Fungsi untuk halaman login
def login_page():
    st.markdown("<h2 style='text-align: center;'>Analisis Sentimen Ganjar Pranowo 2024</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
    
    with st.form(key='login_form', clear_on_submit=True):
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Sign In")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if login_button:
        if username in st.session_state['users'] and st.session_state['users'][username] == password:
            st.success(f"Login berhasil! Selamat datang, {username}.")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username  # Simpan username ke dalam session state
            st.experimental_rerun()
        else:
            st.error("Username atau password salah. Silakan coba lagi.")

    st.markdown("<p style='text-align: center;'>Belum punya akun? <a href='#daftar' onClick='document.getElementById(\"daftar\").scrollIntoView(); return false;'>Daftar</a></p>", unsafe_allow_html=True)
    
    if st.button("Daftar", key="daftar_button"):
        st.session_state['show_register'] = True
    
    if st.session_state.get('show_register'):
        st.markdown("<h2 id='daftar' style='text-align: center;'>Daftar</h2>", unsafe_allow_html=True)
        with st.form(key='register_form'):
            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
            new_username = st.text_input("Username Baru")
            new_password = st.text_input("Password Baru", type="password")
            daftar_button = st.form_submit_button("Daftar")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if daftar_button:
            if new_username in st.session_state['users']:
                st.error("Username sudah ada. Silakan pilih username lain.")
            else:
                st.session_state['users'][new_username] = new_password
                st.success("Pendaftaran berhasil! Silakan login dengan akun baru Anda.")
                st.session_state['show_register'] = False

# Halaman login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_page(st.session_state['username'] == 'admin')
