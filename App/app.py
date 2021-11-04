import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import altair as alt


#Load our pipeline 

pipe_lr = joblib.load(open("models/emotion_classifier.pkl","rb"))


def predict_emotions(text): 
    result=pipe_lr.predict([text])
    return result[0]
def get_predictions_proba(text):
    result=pipe_lr.predict_proba([text])
    return result
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main() : 
    st.title("Emotion Classifier App") 

    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice == "Home" : 
        st.subheader("Home-Emotion In Text")
        with st.form(key='emotion_clf_form'):
            raw_text=st.text_area('Type Here')
            submit_text=st.form_submit_button(label='Submit')
        if submit_text :
            col1,col2 = st.columns(2)
            #Apply our model here   : 
            prediction = predict_emotions(raw_text)
            probability= get_predictions_proba(raw_text)

            with col1 : 
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji=emotions_emoji_dict[prediction]
                st.write("{} : {} ".format(prediction,emoji))
                st.write("Confidence : {}".format(np.max(probability)))
            with col2 : 
                st.success("Prediction Probability")
                st.write(probability)
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)
    elif choice =="Monitor" :
        st.subheader("Monitor App")
    else : 
        st.subheader("About")  




if __name__=='__main__':
    main()
