import pandas as pd

def prepare_data(df):
    df_train_id = []
    df_train_toxic = []
    df_train_neutral = []

    for index, row in df.iterrows():
        references = row[['neutral_comment1', 'neutral_comment2', 'neutral_comment3']].tolist()

        for reference in references:
            if len(reference) > 0:
                df_train_id.append(index)
                df_train_toxic.append(row['toxic_comment'])
                df_train_neutral.append(reference)
            else:
                break

    df = pd.DataFrame({
        'comment_id': df_train_id,
        'toxic_comment': df_train_toxic,
        'neutral_comment': df_train_neutral
    })
    
    df['len'] = df['toxic_comment'].apply(len)
    df = df.sort_values('len').drop(columns=['len'])

    return df