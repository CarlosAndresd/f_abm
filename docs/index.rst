.. Agent Based Model for Opinion Formation documentation master file, created by
   sphinx-quickstart on Fri May 19 19:47:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Agent Based Model for Opinion Formation documentation!
===================================================================

Welcome to the documentation for *Agent Based Model for Opinion Formation*. This is (or hopefully will be) a
self-contained package that allows anyone to simulate and analyse different agent-based opinion formation models.

What is an agent-based model?
-----------------------------

In an agent-based model there is a set of agents or individuals that have certain **behaviour** and **communicate**
with each other in a certain way. Think for example of a classroom. The students are the agents, each has a particular
individual behaviour. They also interact with each other. Their actions are determined by their individual personality
and the interaction with other students.

What is an opinion formation model?
-----------------------------------

It is a model that aims to reproduce at least in part the dynamics of opinion formation, that is, how the opinion of a
population changes over time. There are many types of opinion formation models. Here we focus on agent-based models
where the opinions are unidimensional.

What is an agent-based opinion formation model?
----------------------------------------------

Imagine a population of 100 individuals, and a statement that cannot be proven true or false (for instance: *cooking*
*should be taught in school*). Each agent can *agree* or *disagree* to some extent. For simplicity, assume that the
opinion of each agent can be represented by a single number between -1 and 1. If the opinion is -1 then the agent
completely disagrees, if the opinion is 1 then the agent completely agrees, if the opinion is 0 then the agent is
indifferent and any value in between -1 and 1 represents a degree of agreement or disagreement.

Here each agent has a single opinion, that is why in this model the opinions are unidimensional.

Now, people can speak with each other. If one person tells another person its opinion is equivalent to an agent
'showing' or 'sharing' its opinion with another agent. In this way, at any give time a singe agent can 'perceive' the
opinion of some other agents in the population. When this happens, the agent will most likely change its opinion.

In this way, as agents exchange opinions and change their own opinions, the opinion of the complete population changes.
This is what is called **opinion dynamics**. This is what is studied with this package.

Who is this package for?
------------------------

It is for anyone that is curious about opinion formation. From someone that just now learns what opinion formation is,
to a researcher with extensive knowledge of opinion formation models. The idea is that this package will contain an easy
to use GUI that allows anyone to 'explore' and 'play' with opinion dynamics, functions that can be easily modified and
used, and self-contained tools for analysis of opinion formation.

What can you do now?
____________________

There are many things you can do:

- If you are interested in reading more about the theory behind agent-based opinion formation models, and in particular
the one that inspires the functioning of this package, you can :ref:`read here <theory_page>`.
- If you want to start seeing some simulations you can download the GUI and follow some examples.
- If you want to learn about the functioning of the program and the details of the functions you can go to the
documentation.
- If you want to contribute, even with something as simple as comments to this first introduction feel free to send a
comment here.



.. toctree::
   :maxdepth: 2
   :caption: Theory:



.. include:: modules.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
