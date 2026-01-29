Start: this repo is to be an installable library. Previously i hacked it together. I want to move the files in the proper way to insice src/statflow

1. Remove running the MLflow server functionality, keeping only the URI to the running server
2. Now, let's refactor the home page.
Remove everything below
🗄️ MLflow Server
Ensure the MLflow server is running externally at <http://0.0.0.0:5000> to begin analysis.

Make sure you don't fail on this error.
```python 
2026-01-29 20:21:37.061 Uncaught app execution
Traceback (most recent call last):
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 670, in code_to_exec
    _mpa_v1(self._main_script_path)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 166, in _mpa_v1
    page.run()
    ~~~~~~~~^^
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/navigation/page.py", line 310, in run
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fsx/repos/statflow/src/statflow/Home.py", line 27, in <module>
    from streamlit_comparison.config import (
    ...<6 lines>...
    )
ModuleNotFoundError: No module named 'streamlit_comparison'
2026-01-29 20:21:39.843 Uncaught app execution
Traceback (most recent call last):
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 670, in code_to_exec
    _mpa_v1(self._main_script_path)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 166, in _mpa_v1
    page.run()
    ~~~~~~~~^^
  File "/home/fsx/repos/statflow/.venv/lib/python3.13/site-packages/streamlit/navigation/page.py", line 310, in run
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fsx/repos/statflow/src/statflow/Home.py", line 27, in <module>
    from streamlit_comparison.config import (
    ...<6 lines>...
    )
ModuleNotFoundError: No module named 'streamlit_comparison'
```

---

The goal of our work is to abstract the streamlit app to work with any mlflow experiment.

In home, first, we should, with st.pill, select the experiments that we want to use. for this we need to extract all the experiment names available in the server

---

Add a spinner to get_experiment_names.

they should be multi-select, so we can use multiple experiments instead of just one.

We must make ALL_DATASETS completely dynamic, based on a given (with a dropdown or field) parameter or datasets registered. Make this a function, since we may have to call it in more than one place.

Make this the second step in the process in Home, inside the initial setup expander

Make sure the chosen datasets here will filter the available datasets everywhere else in the webapp.

Elsewhere in the app, we also don't have to query the mlflow server for the datasets available. we should use (and filter) based on the ones we save to the session_state

can we have a sort_items(st.session_state.file_order, key="dataset_order") below (and inside the expander) the dataset pills?
sort_items should automatically update when we select/unselect datasets on the st.pills

we cannot assume the datasetname is "params.dataset_name"; we should ask the user. Firstly, there is already a "Datasets" field, in which we register the dataset, however i do not know how to get it.

the implementation should be:
Before "Select Datasets", we should have:
"If not using the default Datasets, select parameter to define datasets".
For this, we should obtain all the parameters present in all the experiments selected (save this to state so we don't ping the server again for this) and let the user chose one. As default, we should use the MLFLow "Dataset" but i dont know how that is stored;
As a little nice gesture, if we find "dataset" or "dataset" and "name" in the same "params.<here>", assume this is the dataset names

Give a warning "Runs without a value in this field will be filtered out"

Now, after reordering the datasets, let's have another st.expander, this one to deal with parameters. We should obtain all "params." and, in the expander, provide a checkbox to use that parameter for comparison or not

For each param in available_params, we should have, in the same row as the checkbox, a dropdown that reads "Link to other parameter(s)"; these dropdowns should be populated by the existing parameters (minus the one that the dropdown is related to); this functionality will serve to populate the sidebar later on
For a given row, (which should be center vertical align), we should have:
[ ] parameter1 | Link to other parameter | unique parameter value that will be linkeds: [value1, value2, value3]
this means, for example, if we link parameter1 to parameter2's value ABC, that parameter1 will only appear for ABC later on
the dropdown "value to link to" should be multi-select.

only do Running get_experiment_names(). if the server is running

we need to simplify this code. there are a lot of default values being decided everywhere. we need to clean this up.

We should have only one ground truth. we should never try to .get(..., "another default value") from the session_state. if our code is good, we won't have errors. that is our mentality.
Let's clean up the rest of the code base of these .get()

the server is running, but we are getting

Server Not Running

Please ensure the MLflow server is running externally at http://0.0.0.0:5000

this should never be the place where the yaml is stored / checked!!
~/.statflow/config.yaml

it should be wherever the streamlit app is running from, i.e., where it was called in. we must assume it will be called in a repository, so it should be a single .statflow_config.yaml file

There is still the problem:

Server Not Running

Please ensure the MLflow server is running externally at http://localhost:5000

I think we should have ONE mlflow.set_tracking_uri in the code, not many spread out. please fix this and set the mlflow.set_tracking_uri to when we open Home since there is where we will reconnect or connect to different mlflow servers

we should also not have try/excepts in the app; that's bad code behavior because it leads to uncaught bugs.

Let's focus on cleaning up the repository of:

try/except
old, deprecated code
duplicate code


Now, here's how this app should work:

When we first load the experiments, when the user selects the experiments that he wants to load, we should:

load the possible parameters used, as well as metrics, and store them in session_state.
Note: we should never query, anywhere in the app, for possible parameters or metrics. all mlflow server queries will always be to obtain the data itself.
Create a document called AGENTS.md, where you write important notes, for yourself (not human, so don't format it for a human), in the future, so you dont repeat mistakes.

Also:
let's make modular code as much as possible; ideally the pages files should be focused on the streamlit functionality, like UI design and elements; we should have a pages_modules, which contain general modules but also a folder for each created page, for example module_1 which refers to 1_🔬_Single_Dataset.