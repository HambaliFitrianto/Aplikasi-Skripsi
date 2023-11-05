from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import regex as re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# import pickle5 as pickle 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
# from pickle import dump

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">ANALISIS SENTIMEN WISATA API TAK KUNJUNG PADAM MENGGUNAKAN METODE NAIVE BAYES MELALUI ULASAN GOOGLE REVIEW PADA GOOGLE MAPS</h2></center>
""",unsafe_allow_html=True)
#st.write("### Dosen Pengampu: Dr. Cucun Very Angkoso, S.T., MT.",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart', 'check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://storage.nu.or.id/storage/post/16_9/mid/1605569441.JPG" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">dataset tentang ulasan masyarakat terhadap pariwisata api tak kunjung padam dari ulasan google maps. Selanjutnya data ulasan tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma Multinomial Naive Bayes untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)
        st.write(""" 
        <ol>
            <li>Case folding merupakan tahap untuk mengganti keseluruhan kata kapital pada dataset agar berubah menjadi tidak kapital.</li>
            <li>Cleansing yaitu merupakan proses untuk menghilangkan semua simbol, angka, ataupun emoticon yang terdapat didalam dataset</li>
            <li>Slangword Removing yaitu satu proses yang dilakukan untuk mendeteksi dan menghilangkan kata-kata yang tidak baku di dalam dataset</li>
            <li>Tokenization yaitu proses untuk memisahkan suatu kalimat menjadi beberapa kata untuk memudahkan proses stopword.</li>
            <li>Stopword Removing yaitu proses untuk menghilangkan semua kata hubung yang terdapat pada dataset.</li>
            <li>Steaming yaitu proses yang digunakan untuk menghilangkan semua kata imbuhan dan merubahnya menjadi kata dasar.</li>
        </ol> 
        """,unsafe_allow_html=True)
        st.write("#### Dataset")
        df = pd.read_csv("dataprep.csv")
#         df = df.drop(columns=['nama','sentiment','score'])
        st.write(df)
    elif selected == "Implementation":
        #Getting input from user
        word = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            def prep_input_data(word, slang_dict):
                #Lowercase data
                lower_case_isi = word.lower()

                #Cleansing dataset
                clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)

                #Slang word removing
                def replace_slang_words(text):
                    words = nltk.word_tokenize(text.lower())
                    words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
                    for i in range(len(words_filtered)):
                        if words_filtered[i] in slang_dict:
                            words_filtered[i] = slang_dict[words_filtered[i]]
                    return ' '.join(words_filtered)
                slang = replace_slang_words(clean_symbols)

                #Inisialisai fungsi tokenisasi dan stopword
                # stop_factory = StopWordRemoverFactory()
                tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
                tokens = tokenizer.tokenize(slang)

                stop_factory = StopWordRemoverFactory()
                more_stopword = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                                'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                                'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                                '&amp', 'yah']
                data = stop_factory.get_stop_words()+more_stopword
                removed = []
                if tokens not in data:
                    removed.append(tokens)

                #list to string
                gabung =' '.join([str(elem) for elem in removed])

                #Steaming Data
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(gabung)
                return lower_case_isi,clean_symbols,slang,gabung,stem
            
            #Kamus
            with open('combined_slang_words.txt') as f:
                data = f.read()
            slang_dict = json.loads(data)

            #Dataset
            Data_ulasan = pd.read_csv("dataprep.csv")
            ulasan_dataset = Data_ulasan['ulasan']
            sentimen = Data_ulasan['label']

            # TfidfVectorizer 
            tfidfvectorizer = TfidfVectorizer(analyzer='word')
            tfidf_wm = tfidfvectorizer.fit_transform(ulasan_dataset)
            tfidf_tokens = tfidfvectorizer.get_feature_names_out()
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
            
            #Train test split
            # Memisahkan data menjadi training set dan test set
            X_train, X_test, y_train, y_test = train_test_split(tfidf_wm, sentimen, test_size=0.1, random_state=1)

            # model
            # with open('modelpola.pkl', 'rb') as file:
            #     loaded_model = pickle.load(file)
            # clf = loaded_model.fit(X_train,y_train)
            # y_pred=clf.predict(X_test)

            # Membuat instance objek dari kelas MultinomialNB
            clf = MultinomialNB()

            # Melatih model menggunakan data pelatihan
            clf.fit(X_train, y_train)

            # Membuat prediksi menggunakan data testing
            y_pred = clf.predict(X_test)

            #Evaluasi
            # Menghitung akurasi
            akurasi = accuracy_score(y_test, y_pred)

            # Inputan 
            lower_case_isi,clean_symbols,slang,gabung,stem = prep_input_data(word, slang_dict)
            
            # #Prediksi
            v_data = tfidfvectorizer.transform([stem]).toarray()
            y_preds = clf.predict(v_data)

            st.subheader('Preprocessing')
            st.write(pd.DataFrame([lower_case_isi],columns=["Case Folding"]))
            st.write(pd.DataFrame([clean_symbols],columns=["Cleansing"]))
            st.write(pd.DataFrame([slang],columns=["Slang Word Removing"]))
            st.write(pd.DataFrame([gabung],columns=["Stop Word Removing"]))
            st.write(pd.DataFrame([stem],columns=["Steaming"]))
            # st.write('Case Folding')
            # st.write(lower_case_isi)
            # st.write('Cleaning Simbol')
            # st.write(clean_symbols)
            # st.write('Slang Word Removing')
            # st.write(slang)
            # st.write('Stop Word Removing')
            # st.write(gabung)
            # st.write('Steaming')
            # st.write(stem)

            # st.subheader('Akurasi')
            # st.info(akurasi)
            # Mengubah akurasi menjadi persen dengan 2 desimal
            akurasi_persen = akurasi * 100

            # Menampilkan akurasi
            st.subheader('Akurasi')
            st.info(f"{akurasi_persen:.2f}%")

            

            st.subheader('Prediksi')
            if y_preds == "positive":
                st.success('Positive')
            else:
                st.error('Negative')

            # Classification Report
            classification_rep = classification_report(y_test, y_pred)
            st.subheader('Classification Report:\n')
            st.code(classification_rep)

            # # Confusion Matrix Heatmap
            # st.subheader('Confusion Matrix Heatmap')
            # fig, ax = plt.subplots()
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], ax=ax)
            # st.pyplot(fig)

            # Menghitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Menampilkan confusion matrix dengan plot
            st.subheader('Confusion Matrix')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot()

    elif selected == "Tentang Kami":
        st.write("##### Mata Kuliah = Pengenalan Pola - B") 
        st.write('##### Kelompok :')
        st.write("1. Hambali Fitrianto (200411100074)")
        st.write("2. Choirinnisa' Fitria (200411100149)")
