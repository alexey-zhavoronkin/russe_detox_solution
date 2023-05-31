from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        
        self.data = data

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.data)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        comment_id = self.data.loc[index, 'comment_id']
        src_text = self.data.loc[index, 'toxic_comment']
        tgt_text = self.data.loc[index, 'neutral_comment']

        output = {
            'src_text': src_text, 
            'tgt_text': tgt_text, 
        }
        
        return comment_id, output