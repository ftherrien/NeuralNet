Felix Therrien - 10802482
Machine Learning - CSCI575

I used Python 3.5.4 on Fedora 25, however I made sure it works on Windows 10.

To run the code:

$ pyhton3 neural.py

**Given that the "face" folder is in the same directory**

This will read the data, train the network once (about 10 mins) and print the performance using all data (sunglasses + open)

There is only one script containing all the necessary functions and the main program.

The first line of each function explains their behavior. This information can
also be retreived within python3 by doing
>>> import neural as nn
>>> help(nn.function_name)

All the parameters can be modified in the __name__ = __main__ part of the code at the end (line 313 and after).




