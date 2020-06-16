import streamlit as st
import pandas as pd



def main():
    st.title('ABraInBev')

    st.header('Um framework de Deep Learning acessível')
    st.text('Desenvolvido por Alanzera e Murilo Men - Analytics CSC')

    # st.text('Este é um texto com \n quebras de linhas')
    # print('meu print!')

    # st.latex(r'''    a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =   \sum_{k=0}^{n-1} ar^k =    a \left(\frac{1-r^{n}}{1-r}\right)     ''')

    # st.subheader('Colocando uma imagem')
    # st.image(
    #     'data:image/jpeg;base64,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')
    # #st.audio('sample_audio.wav' ) #nao funcionou
    # #st.video('sample_video.mp4')

    # st.markdown('Streamlit is **_really_ cool**.'
    #             '<br> é possível usar $Y=X^2$ '
    #             '<br> na verdade tem a tag de código <code> import streamlit as st</code>'
    #             , unsafe_allow_html=True)  # esse argumento aceita elementos de html para o comentário !

    # # ---------------------- ###
    # st.header('COMANDOS MAIS INTERATIVOS')
    # ###
    # st.subheader('0------------ BOTÃO')
    # botao = st.button('BOTAO')
    # if botao:
    #     st.text('BOTAO CLIKADO COM SUCESSO')
    # ###
    # st.subheader('1------------ CHECK BOX')
    # check = st.checkbox('checkBox')
    # if check:
    #     st.markdown('chackbox selecionado')
    # ###
    # st.subheader('2------------ RADIO')
    # radio = st.radio('SELECIONE O RADIO BUTTON', ('rd 1', 'rd 2'))
    # if radio == 'rd 1':
    #     st.markdown('rd1 selecionado')
    # if radio == 'rd 2':
    #     st.markdown('rd2 selecionado')
    # ###
    # st.subheader('3------------ Select BOX')
    # select = st.selectbox('SELECIONE na SELECT BOX', ('select1', 'select2'))
    # if select == 'select1':
    #     st.markdown('SELECAO 1 selecionado')
    # if select == 'select2':
    #     st.markdown('SELECAO 2 selecionado')
    # ###
    # st.subheader('4------------ MULTISELECT') #alguns problemas

    # multi = st.multiselect('SELECIONE na SELECT BOX', ('ABC', 'DEF','GHI'))
    # if multi == 'ABC':
    #     st.markdown('SELECAO ABC selecionado')
    # if multi == 'DEF':
    #     st.markdown('SELECAO EDF selecionado')
    # ###
    # st.subheader('FILE UPLOADER')
    file = st.file_uploader('Escolha seu arquivo de dados ', type='csv')
    if file is not None:
        # print('sd') - os prints realizados dentro desse script estarão exibidos no console que abri o streamlit run
        # OBJETVO DE PRINTAR O NOME DO ARQUIVO st.markdown('ARQUIVO'+file.__name__+'UPADO')
        st.markdown('ARQUIVO UPADO')
    ############### DF
        st.header('\n Reading DataFrame')

        df = pd.read_csv(file)
        # SLIDER#
        st.subheader('SLIDER')
        slider = st.slider('Quantas linhas você deseja ver ? ', 5, 50, 5)
        # st - dataframe#
        st.subheader('ST DATA FRAME')
        st.dataframe(df.head(slider))
        # st - table

        exploracao = pd.DataFrame({'nomes': df.columns, 'tipos': df.dtypes, 'NA #': df.isna().sum(),
                                   'NA %': (df.isna().sum() / df.shape[0]) * 100})
        st.markdown('** Contagem dos tipos de dados:**')
        st.write(exploracao.tipos.value_counts())
        #explorando
        st.markdown('**colunas float:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))

        st.markdown('**colunas string:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))

        st.markdown('**dados faltantes :**')
        st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])


if __name__ == '__main__':
    main()
