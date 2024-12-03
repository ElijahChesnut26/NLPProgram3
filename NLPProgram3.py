#Eli Chesnut, Tom Kerson, Ani Valluru


from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import random

#function gets the vector for a word in a sentence
def get_word_vec(word, sentence, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state.squeeze(0)
    
    # Find the index of the target word in the tokenized input
    word_idx = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]).index(word)
    
    # Get the vector for the target word
    word_vector = hidden_states[word_idx]#can add [:50] to limit the size of the vector
    return word_vector


def get_df_of_word():
    words = ["overtime", "rubbish", "tissue"]
    overtime_sentences = [r"For its part, management dictated work hours by leaving the implementation of scheduling and overtime compensation policies to the discretion of the project managers,"
        ,r"Thus unlimited work hours without compensation for overtime became a major asset."
        ,r"Paid and unpaid overtime is measured by asking respondents if they had worked paid overtime and if they had worked unpaid overtime in the last two week pay period."
        ,r"The women were only entitled to 4 hours of overtime work per week. "
        ,r"Aside from receiving poverty wages, many caregivers fail to receive legally mandated overtime compensation. "
        ,r"The standards analyzed that require employer payments were minimum wage, overtime, paid time off, unemployment/employment insurance and workers' compensation. "
        ,r"Thus, to force them to work harder, the management introduced ‘obligatory overtime work’ in return for extra days off. "
        ,r"The U.S. ranks significantly better than Canada on: minimum wage, overtime, and occupa tional health and safety. "
        ,r"Was this income part of the remuneration for your regular work, was it payment for overtime, or both?"
        ,r"If you work on a holiday it goes to the time bank and if you agree to work overtime it’s the time bank."
        ,r"The public overtime work that I am proposing will make the champions of misogyny unsafe."
        ,r"They “willingly” oblige to work overtime with no compensation, and the NCAA chooses to look the other way on its own findings."
        ,r"It includes overtime and does not consider any loss of working time due to absenteeism or any other reason."
        ,r"She was unsuccessful both in her attempts to have her overtime recognised "
        ,r"Controller hours were long, the job was stressful, mandatory overtime was common and pay increases were limited by rigid civil service system grids."
        ,r"UNITES has also been active in combating the widespread practice of refusing to pay for overtime. "
        ,r"It may be recalled that quota formulas have evolved overtime and take into account GDP, official reserves, imports and exports."
        ,r"A Norwegian customer once asked me if my husband forced me to work overtime, if he allowed me to make any decisions in the shop, and if he treated me well. "
        ,r"previously everyone had been paid for overtime, but now production workers had to meet targets, based on an estimated daily norm, at a flat rate."
        ,r"While workers are admonished about time efficiencies, and although there is no overtime allowed in most offices, workers must work through until quitting time. "
        ,r"They worked long hours in mostly low-status positions with 95% of women working overtime with limited opportunities for promotion. "
        ,r"I work until midnight, but I obviously don’t have night pay or overtime, and I work Saturdays and Sundays, and that isn’t recognised either, "
        ,r"The latter interventions could include shop stewards monitoring overtime under a collective agreement, or a community’s support for a buy-local programme. "
        ,r"n the best outcome, overtime, this mode of production may become a form of competition that is based on quality rather than quantity."
        ,r"Don't complain about not getting overtime when you work 10 hours a day."
        ,r"Notre Dame had nearly dropped an overtime game to Mississippi, but finished with an unblemished record. "
        ,r"The team ended their losing streak the day before, beating Vegas in overtime. "
        ,r"They are a well-coached group that had DePaul beat but lost in overtime and then lost at home to Bucknell."
        ,r"Pittsburgh's affiliate forced overtime by scoring midway through the final frame. "
        ,r"The running game has done well the last three games, except for a fumble by Tyrone Tracy in overtime. "
        ,r"The Cornhuskers are an overtime loss to Illinois away from being undefeated,"
        ,r"But coming in after the third before the first overtime, we had a lot of confidence in the room."
        ,r"Over the rest of regulation and overtime, they scored 28. "
        ,r"Montreal captain Nick Suzuki scored to open the shootout and Allen shut the door the rest of the way after a scoreless back-and-forth overtime period. "
        ,r"Neither team scored in the remaining time in regulation, leading to the decisive overtime period."
        ,r"He would then score 13 points in overtime to lead Cleveland to a huge win,"
        ,r"Michigan State head coach Tom Izzo reacts to a call during overtime in an NCAA college basketball game against Iowa. "
        ,r"The clock is mostly ornamental in overtime, too. "
        ,r"Additionally, the Iowa product was penalized for holding in overtime."
        ,r"And though they finally claimed victory in overtime, the extra five minutes they required to do so represented another five minutes they could have spared their ailing bodies. "
        ,r"Dallas fought the good fight in their overtime game against the Green Bay Packers last week but wound up with a less-than-desirable result,"
        ,r"It appeared the Mintos had the overtime winner as a puck would go off of both posts, but not find the back of the net."
        ,r"The drama on Sunday started in Germany with the Panthers pulling off an overtime win against the Giants"
        ,r"The Nets were physical and aggressive, and they forced the Celtics into a difficult overtime win."
        ,r"But his fumble to start overtime handed the game away and was unfortunate; he needs to learn from it and move on. "
        ,r"Obviously it's gotten us twice now so, yeah, overtime rules -- not the best. "
        ,r"Both teams had ample scoring opportunities, but the game went to overtime, and eventually PKs. "
        ,r"Allgaier bounced back from an early flat tire and back-to-back penalties, working his way back through the field before passing Austin Hill and Cole Custer on an overtime restart with two laps remaining. "
        ,r"Westhaver scored once in an overtime win for the home team."
        ,r"The Orange is coming off an emotional overtime win over Virginia Tech last week, "
        ]
    word_vectors = [get_word_vec("overtime", sentence, tokenizer) for sentence in overtime_sentences]
    # print(word_vectors[0])
    ot_labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    
    
    pass


def read_file(word):
    


    pass

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

get_df_of_word()