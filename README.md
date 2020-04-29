# multi-index-viz
Visualizes a pandas.Series with a MultiIndex as a Sankey plot

by Michael W. Reimann

This tool generates an interactive visualization of any pandas.Series with the following properties:
1. The value of the Series must be numerical, i.e. of type float or int
2. The Series must be indexed by a pandas.MultiIndex with named columns

The visualization is then a Sankey plot of the strengths of overlap between the values of two columns of the MultiIndex.
Example - our Series object looks like this:

    Fruit         Color   Taste
    apples        red     delicious    1.00
                          yuck         0.10
                  purple  delicious    0.00
                          yuck         0.05
                  green   delicious    0.80
                          yuck         0.20
    grapes        red     delicious    0.20
                          yuck         0.10
                  purple  delicious    1.20
                          yuck         0.30
                  green   delicious    1.50
                          yuck         0.05
    strawberries  red     delicious    2.20
                          yuck         0.00
                  purple  delicious    0.00
                          yuck         2.10
                  green   delicious    0.00
                          yuck         3.00
    dtype: float64

Thus, when choosing the columns "Color" and "Taste", the overlap of "red" and "delicious" would be 1.0 + 0.2 + 2.2 = 3.4,
while the overlap of "red" and "yuck" would be 0.1 + 0.1 + 0.0 = 0.2.
A more complex example, a visualization of all synapses in the mouse-neocortex-connectome of the BBP is deployed under:
https://multi-index-viz.herokuapp.com/

**Getting started**

To start with, let's make a visualization of the fruits example above. The data for this is already part of the repository.

1. Clone the repository from github:

    `git clone https://github.com/MWolfR/multi-index-viz.git`
    
    `cd multi-index-viz`

2. Ensure that your python version matches the specified runtime:

    `cat runtime.txt`

    `python-3.7.3`

    It will probably work in any python-3 version. But only the specified version is officially supported.

3. Install requirements

    `pip install -r requirements.txt`

4. (Optional) install gunicorn
This step is not required, but you can apt-get install gunicorn and then pip install gunicorn

5. Run the example

    `python viz_multi_index.py examples/example-fruits/options_fruits.json`

    `Serving Flask app "viz_multi_index" (lazy loading)
    Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    Debug mode: off
    Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)`


If you have installed gunicorn you can alternatively run:

    gunicorn 'viz_multi_index:main("examples/example-fruits/options_fruits.json")'


    [2020-04-22 09:53:33 +0200] [10516] [INFO] Starting gunicorn 20.0.4
    [2020-04-22 09:53:33 +0200] [10516] [INFO] Listening at: http://127.0.0.1:8000 (10516)
    [2020-04-22 09:53:33 +0200] [10516] [INFO] Using worker: sync
    [2020-04-22 09:53:33 +0200] [10519] [INFO] Booting worker with pid: 10519


Finally, just navigate your browser to the specified url

You can also try out the other examples in the "examples" folder of the repository


**Visualizing your own data**

In the example above, we visualized the data in examples/example-fruits. If we list the contents, we can see that it consists of two files:

    ls examples/example-fruits/
    fruits.json  options_fruits.json

The file "fruits.json" contains the pandas.Series with a MultiIndex, serialized in .json format.
The file "options_fruits.json" contains some configuration data that specifies for example the colors used in the final visualization.

To generate such files for your own data, first serialize your data to a json file.
Remember, that your data must be a pandas.Series with a named MultiIndex. Then do the following:

    In [29]: example_data.head() # Our data is in a pandas.Series object called "example_data"
    Out[29]:
    Fruit   Color   Taste
    apples  red     delicious    1.00
                    yuck         0.10
            purple  delicious    0.00
                    yuck         0.05
            green   delicious    0.80
    dtype: float64

    In [30]: example_data.to_json("examples/my_data.json", orient="table") # orient="table" is important!

Next write an options file associated with your data. To start, you can generate a mostly empty file by using the "default-options" script:

    ./bin/default-options examples/my_data.json

This generates the file examples/options-my_data.json that you can then visualize as specified above:

    python viz_multi_index.py examples/options-my_data.json

---

To change the colors in the plot from a dull grey and to change how labels are presented, you can edit the options file:

Now let's look at the layout of the options file:

    {
      "Data": "fruits.json",
      "Colors": {
        "Hierarchy": ["Color", "Fruit", "Taste"],
        "Values": {red": "rgba(250, 0, 0, 1.0)",
                   [...]
                   "_default": "rgba(120, 120, 120, 1.0)",
                   "ALL": "rgba(0, 0, 0, 1.0)"}
      },
      "Group translations": {
         "Color": "By color",[...]
      },
      "Labelling": {
         "Hierarchy": ["Color", "Taste", "Fruit"],
         "Values": {
           "Color": "{0}",[...]
         }
      },
      "Version": "Example",
      "App":{
        "Default grouping": [[], ["Fruit"], ["Color"], ["Taste"]],
        "Default filter": [[[], []], [[], []], [[], []]],
        "Strength threshold": [0.0, 3.0]
        }
    }

Data: This entry specifies the path to the file holding the pandas.Series, i.e. the file we serialized above.
Alternatively, you can put the _contents_ of that file under this entry, since it is also in .json format. In that case you have everything in a single file!

* Colors: Specifies the colors of nodes in the plot. Consists of two sub-entries:
  * Hierarchy: This must be a permutation of the columns of the MultiIndex. It specifies in which order the levels of the index determine the color of the label.
In the example, if data was grouped by Fruit and Taste, then Fruit would determine the color of the associated label, since it is mentioned first in the list.
  * Values: The actual colors associated with the individual values of the index. Colors must be a string specifying an "rgba" color!
If a value does not exist, instead the value of "_default" is used and that entry must exist! Additionally, an entry called "ALL" must exist!

* Group translations: This is simply a dict that is translating the columns of the MultiIndex into how they are to be labelled in the plot.
If a column has no value in the dict it remains unchanged. This means, the entry "Group translations" has to exist, but it can be an empty dict.

* Labelling: Similar purpose as the above. It determines how the values in the columns show up in the plot.

  * Hierarchy: This must be a permutation of the columns of the MultiIndex. It specifies the order in which the values are used to label the plot.
    The order in the example above would result in for example "red delicious apple" (Color - Taste - Fruit).

  * Values: A dict that specifies how the value of a column is turned into a string. Uses the python string.format function.
    For example, "{0}-colored" would turn the value "green" of the column "Color" into "green-colored". If a value has no value in the dict, then "{0}" is used, i.e. it remains unchanged.
    That means, "Values" has to exist, but it can be an empty dict.

* Version: A string that is used as a title of the plot

* App: Specifies the initial state of the controls of the plot.

  * Default grouping: The plot can be grouped by any column or combination of columns, depending on user input. This entry specifies the initial grouping used in the plot.
    Must be a list of length 4. Each entry of the list is a list of any number of column names, or an empty list.

  * Default filter: The plot can be filtered according to the values of columns. This entry specifies the initial filtering.
    Each filter is a list of length 2, where the first entry is the name of a column and the second is a list of valid values. Any entry where the value of the column is not in the list of valid values is ignored.
    No filtering is instead specified as a list of two empty lists. The entry "Default filter" must be a list of exactly 3 such filters.

  * Strength threshold: The plot can be filtered according to the strength of the overlap where all overlaps below a value are ignored.
    This entry is a list of length 2 specifying the minimal and maximal selectable value of that filtering.
  
  * Height (optional): The height of the main plot in pixels. Default value: 700
  
  * Plot type (optional): How the data is actually visualized. The default value is "Sankey", which leads to a Sankey plot.
    Other options are "Starburst", which leads to a starburst plot and "Bar" which leads to bar plots that are vertically grouped and horizontally stacked.
  
  * Filter control types (optional): Determines how filtering is performed. The default value is "Dropdown", which leads to a dropdown menu from which the user can select one or multiple valid values.
    Alternative: "RangeSlider", which leads to a slider from which the user selects a minimum and maximum value. If this is used in conjunction with non-numerical data, no filtering will be applied.
    

