""" beyond offering a list of keywords, the definitive labelling of a topic cluster is 
    a difficult problem, at least without access to partially-labelled training data.  
    An unsupervised learning approach to this task is to integrate powerful pre-trained 
    models such as ChatGPT, or large knowledge graphs such as ConceptNET, through their webAPI """


import openai
from num2words import num2words
from api_keys import secret


def chatGPT_cluster_label(articles):
    """ uses a list of titles and abstracts as basis for a chatgpt query """
    openai.api_key = secret.key
    completion = openai.ChatCompletion()
    length = len(articles)

    # convert articles list into string with XML tags
    articles=["<article>"+ article +" </article> \n" for article in articles]
    articles = "".join(articles)
    
    # create query
    query = [{'role'   : 'system',
              'content': f"""You are an intelligent but laconic robot that responds by using the fewest words possible, at all times. 
                             Your task is to assign name labels to groups of short encyclopedia articles that are clustered by topic.
                             You will be provided with {num2words(length+1)} articles (delimited with XML tags) that are selected at random from the topic cluster. 
                             Your cluster label must name the group, or set, of which all the articles are a member.  
                             Your cluster label must be a maximum of three words long, and ideally only one or two words, but also capture as much fine detail about the cluster's topic as possible."""},
             {'role'   : 'user', 
              'content': f'Hello robot, here are the {num2words(length+1)} articles: \n{articles}\n What is your cluster label?'}]
    
    # get response
    response = completion.create(model='gpt-3.5-turbo', messages=query)
    answer = response.choices[0]['message']['content']
    return answer

