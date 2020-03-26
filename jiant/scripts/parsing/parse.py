""" Rohit Roongta """

import json
import os

"""
Sample data
{"idx": 0, "passage": { "questions": [{"idx": 0, "answers": [{"idx": 0, "label": 1}, {"idx": 1, "label": 1}, {"idx": 2, "label": 1}, {"idx": 3, "label": 1}]}, {"idx": 1, "answers": [{"idx": 4, "label": 1}, {"idx": 5, "label": 1}, {"idx": 6, "label": 1}, {"idx": 7, "label": 1}, {"idx": 8, "label": 1}]}, {"idx": 2, "answers": [{"idx": 9, "label": 1}, {"idx": 10, "label": 1}, {"idx": 11, "label": 1}, {"idx": 12, "label": 1}]}, {"idx": 3, "answers": [{"idx": 13, "label": 1}, {"idx": 14, "label": 1}, {"idx": 15, "label": 1}, {"idx": 16, "label": 1}, {"idx": 17, "label": 1}, {"idx": 18, "label": 1}, {"idx": 19, "label": 1}]}]}}
"""

""" Change the file name """
input_file = 'MultiRC_val.jsonl'
output_file = 'expectedData.jsonl'

with open(input_file) as file:

    """ Delete the output file if exists """
    if os.path.exists(output_file):
        os.remove(output_file)
    
    """ read line by line """
    lines = file.readlines()
    for line in lines:
    
        """ strips remove the new line character """
        json_str = line.strip()
        
        """ loads converts string to json [has (s)] whereas load function loads a file """
        data = json.loads(json_str)
        parse_data = {}
        
        """ Add the passage index tag and value """
        parse_data["idx"] = data['idx']
        
        """ Create passage dictionary """
        parse_data["passage"] = {}
        
        passage_data = data['passage']
        
        """ Create question list """
        questions = []
        
        """ Traversing through the passage questions """
        for question in passage_data['questions']:
            
            """ Create question dictionary """
            question_dict = {}
            
            """ Add the question index to the question dictionary """
            question_dict["idx"] = question['idx']
            
            """ Create answer list """
            question_dict["answers"] = []
            answers = question['answers']
            
            """ Traversing through the answers option for the particular question """
            for answers_details in answers:
            
                """ Add the answer index and label to the answer dictionary """
                ans_dict = {}
                ans_dict["idx"] = answers_details['idx']
                ans_dict["label"] = answers_details['label']
                
                """ Append the answer dictionary to question dictionary """
                question_dict["answers"].append(ans_dict)
            
            """ Append the question and answer set to question list """
            questions.append(question_dict)
            
        """ Initialise question key with question list """
        parse_data["passage"]["questions"] = questions
        
        
        """ Writing data to a file """
        with open(output_file, 'a+') as output:
            output.write(json.dumps(parse_data)+"\n")
            
        """ Converting dictionary into string """
        #print(json.dumps(parse_data))
