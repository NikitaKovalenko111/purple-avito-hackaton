import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel


class Config:
    
    is_train = True
    target = 'targetSplitMcIds'

    mc_ids = [108, 101, 109, 102, 106, 107, 111, 110, 105, 104, 103]
    mc_titles = {
        np.int64(108): 'Штукатурные работы',
        np.int64(101): 'Ремонт квартир и домов под ключ',
        np.int64(109): 'Напольные покрытия',
        np.int64(102): 'Сантехника',
        np.int64(106): 'Поклейка обоев',
        np.int64(107): 'Малярные работы',
        np.int64(111): 'Демонтажные работы',
        np.int64(110): 'Гипсокартон',
        np.int64(105): 'Укладка плитки',
        np.int64(104): 'Натяжные потолки',
        np.int64(103): 'Электрика'
    }

    offer_selected_features = ["sourceMcTitle", "description"]

    Bert_parameters = {
        "best_model_name": 'google/bert_uncased_L-4_H-768_A-12'
    }


class Dataset:
    


    def __init__(self, config: Config):
        self.config = config
    
    def preprocessing(self, df: pd.Dataframe):
        
        new_df = df[self.config.offer_selected_features]

        return new_df


class MiniBert(nn.Module):

    def __init__(self, bert_model_name='google/bert_uncased_L-4_H-768_A-12', output_dim=256, freeze_bert=True):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, text_inputs):
        
        if isinstance(text_inputs, list):
            encoded = self.tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            input_ids = encoded["input_ids"].to(self.bert.device)
            attention_mask = encoded["attention_mask"].to(self.bert.device)
        
        else:

            input_ids, attention_mask = text_inputs
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        offer_embedding = self.projection(cls_embedding)

        return offer_embedding


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss, где эмбеддинги классов передаются как параметр.
    """
    def __init__(self, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, user_embs, class_embs, labels):
        """
        user_embs: [batch_size, D] – эмбеддинги объектов
        class_embs: [num_classes, D] – эмбеддинги классов
        labels: [batch_size] – индексы правильных классов
        """
        # Нормализуем эмбеддинги
        user_embs = F.normalize(user_embs, p=2, dim=1)   # [batch, D]
        class_embs = F.normalize(class_embs, p=2, dim=1) # [num_classes, D]

        # Косинусы между каждым объектом и каждым классом
        cosine = torch.matmul(user_embs, class_embs.T)    # [batch, num_classes]

        # Косинус для правильного класса
        cos_target = cosine[torch.arange(len(labels)), labels].unsqueeze(1)

        # sin(θ) = sqrt(1 - cos²θ) с защитой от погрешностей
        cos_target_clamped = torch.clamp(cos_target, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_target = torch.sqrt(1.0 - cos_target_clamped ** 2)

        # cos(θ + m)
        cos_new = cos_target_clamped * torch.cos(self.margin) - sin_target * torch.sin(self.margin)

        # Логиты: для всех классов масштабируем косинусы, для правильного заменяем
        logits = cosine * self.scale
        logits.scatter_(1, labels.unsqueeze(1), cos_new * self.scale)

        # Кросс-энтропия
        loss = F.cross_entropy(logits, labels)
        return loss


class SoftmaxModel(nn.Module):


	def __init__(self, offer_features_dim, embedding_dim, num_items):

            super().__init__()
            
            self.arcface_loss = ArcFaceLoss()
            self.encoder = MiniBert()
            self.num_items = num_items

        def forward(self, offer_features, labels=None):
                mc_titles = {
                    np.int64(108): 'Штукатурные работы',
                    np.int64(101): 'Ремонт квартир и домов под ключ',
                    np.int64(109): 'Напольные покрытия',
                    np.int64(102): 'Сантехника',
                    np.int64(106): 'Поклейка обоев',
                    np.int64(107): 'Малярные работы',
                    np.int64(111): 'Демонтажные работы',
                    np.int64(110): 'Гипсокартон',
                    np.int64(105): 'Укладка плитки',
                    np.int64(104): 'Натяжные потолки',
                    np.int64(103): 'Электрика'
                }

                mc_titles = torch.tensor(list(mc_titles.values()))

                mc_embs = self.encoder(mc_titles)
                offer_embs = self.encoder(offer_features["description"])

                if labels != None:
                    loss = self.arcface_loss(offer_embs, mc_embs, labels)
                    return loss
                
                else:
                    mc_embs = F.normalize(mc_embs, p=2, dim=1)
                    offer_embs = F.normalize(offer_embs, p=2, dim=1)
                    logits = torch.matmul(mc_embs, offer_embs.T)

                    return logits