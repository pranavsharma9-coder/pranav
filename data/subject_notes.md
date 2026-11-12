#Definitions of Machine Learning 
Machine Learning is the science (and art) of programming computers so they 
can learn from data. 
Machine Learning is the field of study that gives computers the ability to learn 
without being explicitly programmed. 
A computer program is said to learn from experience E with respect to some task 
T and some performance measure P, if its performance on T, as measured by P, 
improves with experience E. 
Need to use Machine Learning 
Machine Learning is great for: 
Problems for which existing solutions require a lot of hand-tuning or long lists of 
rules: one Machine Learning algorithm can often simplify code and perform better. 
Eg: writing traditional programming vs machine learning approach for a spam filte
In traditional programming, one writes a detection algorithm for each of the patterns that 
you noticed, and the program would flag emails as spam if a number of these patterns are 
detected. Since the problem is not trivial, your program will likely become a long list of 
complex rules—pretty hard to maintain. In contrast, a spam filter based on Machine 
Learning techniques automatically learns which words and phrases are good predictors 
of spam by detecting unusually frequent patterns of words in the spam examples 
compared to the ham examples. The program is much shorter, easier to maintain, and 
most likely more accurate. 
Complex problems for which there is no good solution at all using a traditional 
approach: the best Machine Learning techniques can find a solution. 
Machine Learning shines is for problems that either are too complex for traditional 
approaches or have no known algorithm. For example, consider speech recognition: 
say you want to start simple and write a program capable of distinguishing the words 
“one” and “two.” You might notice that the word “two” starts with a high-pitch sound 
(“T”), so you could hardcode an algorithm that measures high-pitch sound intensity 
and use that to distinguish ones and twos. Obviously this technique will not scale to 
thousands of words spoken by millions of very different people in noisy environments 
and in dozens of languages. The best solution (at least today) is to write an algorithm 
that learns by itself, given many example recordings for each word. 
Fluctuating environments: a Machine Learning system can adapt to new data. 
Eg: In the spam filter detection example, if spammers notice that all their emails 
containing “4U” are blocked, they might start writing “For U” instead. A spam filter 
using traditional programming techniques would need to be updated to flag “For U” 
emails. If spammers keep working around your spam filter, you will need to keep 
writing new rules forever. In contrast, a spam filter based on Machine Learning 
techniques automatically notices that “For U” has become unusually frequent in spam 
flagged by users, and it starts flagging them without your intervention
In batch learning, the system is incapable of learning incrementally: it must be trained 
using all the available data. First the system is trained, and then it is launched into 
production and runs without learning anymore; it just applies what it has learned. This 
is called offline learning. 
Drawbacks: 
Handling large amounts of data: Batch learning requires loading the entire dataset into 
memory for training. This becomes a challenge when dealing with large datasets that 
exceed the available memory capacity. 
Hardware limitations: Batch learning can be computationally expensive, especially when 
dealing with complex models or large datasets. Training a model on a single machine may 
take a significant amount of time and may require high-performance hardware, such as 
GPUs or specialized processing units. 
Availability constraints: In some scenarios, obtaining the entire dataset required for batch 
learning may not be feasible or practical. 
online learning  
In the online learning, data is fed to the model in small batches, sequentially. These batches 
are called mini batches. After, each batch of training, your model gets better. since these 
batches are small chunks of data. so you can perform this training on server (in 
production) That’s why it is called online learning means your model is getting trained 
when your model is on server. 
Advantages of Model-Based Learning 
1. Faster predictions: Model-based learning is typically faster than instance
based learning because the model is already created and can be used to make 
predictions quickly. 
2. More accurate predictions: Model-based learning can often make more 
accurate predictions than instance-based learning because the model is 
trained on a large dataset and can generalize to new data. 
3. Better understanding of data Model-based learning allows you to gain a better 
understanding of the relationships between input and output variables.  
Disadvantages of Model-Based Learning 
1. Requires a large dataset: model-based learning requires a large dataset to 
train the model.  
2. Requires expert knowledge: Model-based learning requires expert knowledge 
of statistical algorithms and mathematical modeling. 
3. Requires expert knowledge: Model-based learning requires expert knowledge 
of statistical algorithms and mathematical modeling. 
Advantages of Instance-Based Learning 
1. No need for model creation: Instance-based learning doesn’t require creating 
a model. 
2. Can handle small datasets: Instance-based learning can handle small datasets 
because it doesn’t require a large dataset to create a model. 
3. More flexibility: Instance-based learning can be more flexible than model
based learning because the machine stores all instances of data and can use 
this data to make predictions. 
Disadvantages of Instance-Based Learning 
1. Slower predictions: Instance-based learning is typically slower than model
based learning because the machine has to compare the new data to all 
instances of data in order to make a prediction. 
2. Less accurate predictions: Instance-based learning can often make less 
accurate predictions than model-based learning because it doesn’t have a 
mathematical model to generalize from. 
3. Limited understanding of data: Instance-based learning doesn’t provide as 
much insight into the relationships between input and output variables as 
model-based learning does.