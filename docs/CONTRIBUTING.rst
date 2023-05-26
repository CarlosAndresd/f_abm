Guideline on how to contribute
==============================

First of all, thanks for wanting to contribute! We hope you can find all
the information you are looking for, regarding contributions, in this
document. If not, feel free to suggest a modification via a pull
request.

Ways to contribute
------------------

There are many ways to contribute,

-  if you don’t want to read or write code you may `have a question or
   suggestion <#you-have-a-question-or-suggestion>`__, or have an idea
   for `new functionality <#you-have-an-idea-for-new-functionality>`__,
-  if you only want to read code you may `contribute with
   documentation <#you-want-to-contribute-with-documentation>`__ either
   checking or writing docstrings, tutorials, or information files,
-  if you want to use the existing code you may want to `add examples or
   tests <#you-want-to-add-examples-or-tests>`__
-  if you feel like contributing to the existing code you may want to
   `write or improve some
   code <#you-want-to-write-or-improve-some-code>`__ or `implement some
   functionality <#you-want-to-implement-new-functionality>`__,
-  or, maybe you were using the program and you `have found a
   bug <#you-have-found-a-bug>`__ and wish to let us know.

You have a question or suggestion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Great! You are helping us create a more clear and improved repository.
Simply raise a `corresponding
issue <https://github.com/CarlosAndresd/f_abm/issues/new?assignees=&labels=question&template=questions-or-suggestions.md&title=Question_Suggestion>`__

You have found a bug
~~~~~~~~~~~~~~~~~~~~

Please, let us know what happened with the `corresponding
issue <https://github.com/CarlosAndresd/f_abm/issues/new?assignees=&labels=bug&projects=&template=bug_report_template.yml&title=Bug+Report>`__.

You want to write or improve some code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thanks! when writing new code please keep in mind the `following
guidelines <#guidelines-when-adding-new-code>`__

You want to add examples or tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thanks! Please follow the `guidelines to add new
code <#guidelines-when-adding-new-code>`__, and in addition to that do
the following:

-  If you are adding tests, make sure that they are run automatically.
-  If you are adding examples, please document well the examples and the
   use of functions in them.
-  In both cases, make sure to add the tests or examples in the correct
   directory.

You have an idea for new functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All new ideas are welcome, please provide details for your idea with the
`following
issue <https://github.com/CarlosAndresd/f_abm/issues/new?assignees=&labels=enhancement&projects=&template=new_functionality.yml&title=New+Functionality>`__

You want to implement new functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thanks! Please follow the `guidelines to add new
code <#guidelines-when-adding-new-code>`__. If the functionality was
suggested in an issue, please make sure to reply or close the issue. If
the functionality was not suggested in an issue, please make an issue
and then reply or close it.

You want to contribute with documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Great!

You can contribute to the documentation in a number of ways:

-  reading the current documentation and making sure that it is clear
   and error-free;
-  adding diagrams or figures when you feel can make the explantion more
   clear;
-  adding explanations on theoretical parts of the documentation;
-  making sure that all functions and modules have complete docstrings.

In this repository documentation for the code is made automatically
using Sphinx and HTML webpages, so there is no need to manually document
the code in the HTML webpage.

Guidelines when adding new code
===============================

-  The prefered docstrings style is `Numpy-Style
   Docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`__.
-  Follow the code style guidelines of
   `PEP8 <https://peps.python.org/pep-0008/>`__. This is easily done if
   you write code using an IDE like
   `PyCharm <https://www.jetbrains.com/pycharm/>`__
-  When defining new functions and variables, please use the
   `PEP8 <https://peps.python.org/pep-0008/>`__ standard. In a nutshell:

   -  Function names should be lowercase.
   -  Variable names should be lowercase.

-  Whenever possible, add in the function docstrings a simple example of
   how to use the function. Also, make use of default parameters to be
   able to use the functions in a simple way.
-  When adding functions to a module remenber to list them in the
   docstrings of the module.
-  When importing functions from other modules explicitly mention them
   in the import statement.
