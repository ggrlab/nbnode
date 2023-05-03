import copy
import inspect
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anytree
import datatable
import matplotlib
import numpy as np
import pandas as pd
import pydotplus
from anytree.exporter import DotExporter
from matplotlib.colors import to_hex

from nbnode.plot.shifted_colormap import shifted_colormap

# Set matplotlib backend to "AGG"
matplotlib.use("AGG")


class NBNode(anytree.Node):
    """Non-binary node class, inherits from anytree.Node."""

    def __init__(
        self,
        name: str,
        parent: "NBNode" = None,
        decision_value: Any = None,
        decision_name: str = None,
        decision_cutoff: float = None,
        **kwargs,
    ):
        """Non-binary node class, inherits from anytree.Node.

        Args:
            name (str):
                The name of the node. Should be unique within the children of a
                parent node.
            parent (NBNode, optional):
                The parent node. Defaults to None.
            decision_value (Any, optional):
            The value leading to this node.
                Can be anything, including single values, strings, numbers, lists,
                dicts, etc. NBNode uses exact equality to determine which child node to
                take.

                Defaults to None.
            decision_name (str, optional):
                The name of the value leading to this node. Defaults to None.
            decision_cutoff (float, optional):
                The cutoff value for the ``decision_value``. If the ``decision_value``
                is numeric, then the ``decision_value`` is compared to the
                decision_cutoff and returns::

                    1       if value >= cutoff
                    -1      if value < cutoff
                    value   if cutoff is None

                Then ``decision_value`` should then be either 1 or -1 such that
                after the comparison, the ``decision_value`` is either 1 or -1.
                Defaults to None.
        """
        super(NBNode, self).__init__(name=name, parent=parent, **kwargs)
        self.parent = parent
        self.decision_name = decision_name
        self.decision_value = decision_value
        self.decision_cutoff = decision_cutoff  # Not necessary in ternary prediction
        self.counter = 0
        self.ids = []
        self._data: pd.DataFrame = None
        self.id_unique_dot_exporter = None
        # Set math_node_attribute to whatever you want to add when using
        # (usual) mathematics.
        self.math_node_attribute = "counter"
        self.math_inplace = False
        self._long_print_attributes = ["counter", "decision_name", "decision_value"]

    def prediction_str(self, nodename: str, split: str = "/") -> "NBNode":
        """Return the node that is the prediction for the given nodename.

        Args:
            nodename (str):
                The name of the node to predict. Should be something matching to any
                ``node.get_name_full()``. You have to start with "/" as the root node.
            split (str, optional):
                The string to split the single node names.
                Defaults to "/". E.g. "/child1/child2" corresponds to the node in the
                hierarchy::

                    root
                    |---child1
                    |   |---child2

        Returns:
            NBNode: The node for the given nodename.
        """
        if isinstance(nodename, str):
            split_names = nodename.split(split)
            # if first element is "", then because that was the root.
            if split_names[0] == "":
                split_names = split_names[1:]
        else:
            split_names = nodename

        if split_names[0] != self.name:
            raise ValueError(
                "self.name != split_names[0]. Did you "
                + "start at (=supply) the right node?"
            )
        if len(self.children) != 0 and len(split_names) > 1:
            for child in self.children:
                if child.name == split_names[1]:
                    # "Remove" [1:] the first element (the current node's name)
                    return child.prediction_str(split_names[1:])
        else:
            return self

    @staticmethod
    def do_cutoff(
        value: Union[float, Any], cutoff: Union[float, None]
    ) -> Union[Literal[1, -1], Any]:
        """If cutoff is not None, cut the value into 1 or -1.

        Otherwise return the value.

        Args:
            value (float): The value to be cut
            cutoff (Union[float, None]): The value to cut at. If None, return value

        Returns:
            Union[Literal[1, -1], Any]:
                1       if value >= cutoff
                -1      if value < cutoff
                value   if cutoff is None
        """
        # ">= --> 1" from https://en.wikipedia.org/wiki/Decision_tree_learning
        if cutoff is None:
            return value
        elif value >= cutoff:
            return 1
        else:
            return -1

    def single_prediction(
        self,
        values: Union[List, Dict],
        names: list = None,
        allow_unfitting_data: bool = False,
        allow_part_predictions: bool = False,
    ) -> Union["NBNode", List["NBNode"]]:
        """Predicts the endnode (leaf) of the tree given the values.

        Args:
            values:
                Either a list or a dict of values. If a dict is given, the keys of
                the dict are used as names. This is used to identify the correct _exact_
                value for the decision node defined by ``self.decision_value``.
            names:
                If values is a list, names is a list of the names of the values.
                This is used to identify the correct value for the decision node defined
                by ``self.decision_name``.
            allow_unfitting_data:
                If True, returns None if the data you gave was not possible to fit in
                the tree. If False, raises a ValueError.
                Useful if decision values only fit partly to the tree but perfectly
                (completely) to another branch of the tree.
            allow_part_predictions:
                If True, returns all (potentially multiple!) nodes that fit the given
                values. They do not have to be leaf nodes.
                If False, returns only the first node that fits the given values.

        Returns:
            Either a single NBNode instance (the leaf node) or if multiple leaf nodes
            fit, all of them as a list.
        """

        if isinstance(values, dict):
            names = list(values.keys())
            values = list(values.values())
        elif isinstance(values, pd.Series):
            names = list(values.index)
            values = list(values)

        leaf_nodes_list = []
        leaf_node = None
        if len(self.children) != 0:
            for child in self.children:
                child: NBNode
                if not isinstance(child.decision_name, list):
                    name_index = names.index(child.decision_name)
                    if child.decision_cutoff is not None:
                        values[name_index] = self.do_cutoff(
                            value=values[name_index], cutoff=child.decision_cutoff
                        )
                    do_all_values_match = values[name_index] == child.decision_value
                else:
                    name_index = [names.index(x) for x in child.decision_name]
                    if child.decision_cutoff is not None:
                        for decision_i, decision_name_i in enumerate(name_index):
                            values[decision_name_i] = self.do_cutoff(
                                value=values[decision_name_i],
                                cutoff=child.decision_cutoff[decision_i],
                            )
                    do_all_values_match = all(
                        [
                            values[decision_name]
                            == child.decision_value[decision_value_index]
                            for decision_value_index, decision_name in enumerate(
                                name_index
                            )
                        ]
                    )
                if do_all_values_match:
                    leaf_node = child.single_prediction(
                        values=values,
                        names=names,
                        allow_unfitting_data=allow_unfitting_data,
                        allow_part_predictions=allow_part_predictions,
                    )
                    if isinstance(leaf_node, list):
                        leaf_nodes_list += leaf_node
                    else:
                        leaf_nodes_list += [leaf_node]
        else:
            return self

        if leaf_node is None:
            if not (allow_unfitting_data or allow_part_predictions):
                raise ValueError(
                    "Could not find a fitting endnode for the data you gave. "
                    + "You also did not allow for part predictions. "
                )
            elif allow_part_predictions:
                leaf_nodes_list += [self]
                return leaf_nodes_list
            else:  # Then only allow_unfitting_data was True
                return leaf_nodes_list
        # if leaf_node is None and not allow_unfitting_data:
        #     raise ValueError(
        #         "Could not find a fitting endnode for the data you gave."
        #         )
        else:
            if len(leaf_nodes_list) == 1:
                leaf_nodes_list = leaf_nodes_list[0]
            return leaf_nodes_list

    def predict(
        self,
        values: Union[List, Dict, pd.DataFrame] = None,
        names: list = None,
        allow_unfitting_data: bool = False,
        allow_part_predictions: bool = False,
    ) -> Union["NBNode", List["NBNode"], pd.Series]:
        """See ``single_prediction``.

        But you can put in dataframes or ndarrays instead of only
        dict + value/key paired lists.

        If values is not given or None, the self._data is used.

        Returns:
            List[NBNodes]: Returns for each value its NBNode
        """
        if values is None:
            if self.data is None:
                raise ValueError(
                    "predict() without argument (the data to predict) is only "
                    + "possible if self._data is not None"
                )
            else:
                values = self._data

        if isinstance(values, datatable.Frame):
            values = values.to_pandas()

        if isinstance(values, np.ndarray):
            if names is None:
                raise ValueError(
                    "You supplied a numpy array but no names. "
                    + "Names are necessary with np.ndarray"
                )
            if len(values.shape) == 1:
                values.shape = (1, len(values))
            values = pd.DataFrame(values, columns=names)

        if isinstance(values, dict) or (
            names is not None and len(values) == len(names)
        ):
            return self.single_prediction(
                values=values,
                names=names,
                allow_unfitting_data=allow_unfitting_data,
                allow_part_predictions=allow_part_predictions,
            )
        elif isinstance(values, pd.DataFrame):
            return values.apply(
                func=self.single_prediction,
                axis=1,
                allow_unfitting_data=allow_unfitting_data,
                allow_part_predictions=allow_part_predictions,
            )
        else:
            # THen I assume I got multiple instances to predict.
            raise ValueError("I do not know how to fit the given inputdata.")

    def reset_counts(self):
        """Set all node.counters to 0."""
        self.counter = 0
        for child in self.children:
            if child is not None:
                child.reset_counts()

    def count(
        self,
        node_list: List["NBNode"] = None,
        reset_counts: bool = True,
        use_ids: bool = False,
    ) -> None:
        """Count ids and save into node.counter.

        Args:
            node_list:
                The [usually predicted] nodes. Usual workflow would be:
                    1. Predict n samples
                    2. Get a list of n nodes from these predictions

                But you can insert here any node (inside the tree) you want.
            reset_counts:
                Should all .counter be set to 0?
            use_ids:
                If use_ids==True, do not use node_list to count but just access the
                length of the node.ids
        """
        if reset_counts:
            self.reset_counts()

        if use_ids:
            for node in anytree.iterators.PreOrderIter(self):
                node.counter = len(node.ids)
        else:
            counted_nodes = Counter(node_list)
            for key_node, node_count in counted_nodes.items():
                ancestor_nodes = list(key_node.ancestors)
                for node_x in [key_node] + ancestor_nodes:
                    node_x.counter += node_count

    def reset_ids(self):
        """Set all node.ids to []."""
        self.ids = []

        for child in self.children:
            if child is not None:
                child.reset_ids()

    def id_preds(self, node_list: List["NBNode"], reset_ids: bool = True):
        """Predict node ids.

        Given a list of nodes, enumerate through them and assign this
        (`enumerate(node_list)`) number to self.ids.

        This is then used to subset self.data for each node.

        Args:
            node_list (List['NBNode']): _description_
            reset_ids (bool, optional): _description_. Defaults to True.
        """
        if reset_ids:
            self.reset_ids()
        for index, node in enumerate(node_list):
            ancestor_nodes = list(node.ancestors)
            for node_x in [node] + ancestor_nodes:
                node_x.ids += [index]

    @property
    def data(self) -> pd.DataFrame:
        """Data of a node for its ids.

        root._data contains all data. However, each node only "holds" a subset of the
        data. To not have to copy the data for each node, we just subset the data
        for each node by the node's ids.

        Usually you would set the ids by `celltree.id_preds(predicted_nodes)`.
        You can also set them manually, but you have to be certain that they match to
        the order of the data!

        Returns:
            pd.DataFrame:
                A subset of the root._data corresponding to the node's ids.
        """
        if self.root._data is None:
            return None

        if self.ids == []:
            warnings.warn(
                "self.ids was an empty list, subset an empty dataframe. Did you call "
                + "celltree.id_preds(predicted_nodes)?"
                + " Can also be a node with no cells."
            )

        return self.root._data.iloc[self.ids, :]

    @data.setter
    def data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            warnings.warn(
                "data is no pandas.DataFrame, converting it via pd.DataFrame(data)."
            )
            data = pd.DataFrame(data=data)
        self.root._data = data

    def apply(
        self,
        fun,
        input_attribute_name: str = "data",
        result_attribute_name: str = None,
        iterator=anytree.iterators.PreOrderIter,
        *fun_args,
        **fun_kwargs,
    ) -> Optional[Dict["NBNode", Any]]:
        """Apply the given function to the `.data` property of each node.

        Args:
            fun: Function to apply on the attribute named `input_attribute_name`
            input_attribute_name: Name of the attribute to apply `fun` on.
            result_attribute_name:
                If result_attribute_name is given, the return value of `fun(node.data)`
                is set to the node's `result_attribute_name`-attribute
            iterator:
                How to iterate over the nodes.
        """
        applied_fun_dict = {}
        for node in iterator(self):
            applied_fun = fun(
                getattr(node, input_attribute_name), *fun_args, **fun_kwargs
            )
            if result_attribute_name is not None:
                setattr(node, result_attribute_name, applied_fun)
            else:
                applied_fun_dict.update({node: applied_fun})

        if len(applied_fun_dict) > 0:
            return applied_fun_dict

    def export_dot(self, unique_dot_exporter_kwargs: Dict = "default") -> str:
        """Convenience wrapper around anytree.DotExporter.

        Args:
            unique_dot_exporter_kwargs (Dict, optional):
                Arguments to anytree.DotExporter.
                Defaults to "default"::

                    unique_dot_exporter_kwargs = {
                        "options": ['node [shape=box, style="filled", color="black"];'],
                        "nodeattrfunc":
                            lambda node: 'label="{}", fillcolor="white"'.format(
                            node.name
                        ),
                    }

        Returns:
            str: A string with the exported dot graph in dot format
        """
        if self.id_unique_dot_exporter is None:
            self.set_DotExporter_ids()
        if unique_dot_exporter_kwargs == "default":
            unique_dot_exporter_kwargs = {
                "options": ['node [shape=box, style="filled", color="black"];'],
                "nodeattrfunc": lambda node: 'label="{}", fillcolor="white"'.format(
                    node.name
                ),
            }
        # dot_data = DotExporter(self, **unique_dot_exporter_kwargs)
        dot_data = DotExporter(
            self,
            nodenamefunc=lambda node: node.id_unique_dot_exporter,
            **unique_dot_exporter_kwargs,
        )
        dotdata_str = "\n".join([x for x in dot_data])
        return dotdata_str

    def set_DotExporter_ids(self):
        """Create unique ids for each node.

        DotExporter needs unique ids for each node. I set them to the
        hex(id(node)) to make sure they are unique.

        """
        for node in anytree.iterators.PreOrderIter(self):
            node.id_unique_dot_exporter = hex(
                id(node)
            )  # That is how DotExporter gets his values

    def graph_from_dot(
        self: "NBNode",
        tree: "NBNode" = None,
        exported_dot_graph: str = None,
        title: str = None,
        fillcolor_node_attribute: str = "counter",
        custom_min_max_dict: Dict[str, float] = None,
        minmax: str = "equal",
        fillcolor_missing_val: str = "#91FF9D",
        node_text_attributes: Union[List[str], Dict[str, str]] = "default",
        cmap: str = matplotlib.cm.RdBu_r,
    ):
        """See `NBNode._graph_from_dot`.

        If no ``tree`` is given, self is used.
        """
        return self._graph_from_dot(
            tree=tree if tree is not None else self,
            exported_dot_graph=exported_dot_graph,
            title=title,
            fillcolor_node_attribute=fillcolor_node_attribute,
            custom_min_max_dict=custom_min_max_dict,
            minmax=minmax,
            fillcolor_missing_val=fillcolor_missing_val,
            node_text_attributes=node_text_attributes,
            cmap=cmap,
        )

    @staticmethod
    def _graph_from_dot(
        # def graph_from_dot(
        tree: "NBNode",
        exported_dot_graph: str = None,
        title: str = None,
        fillcolor_node_attribute: str = "counter",
        custom_min_max_dict: Dict[str, float] = None,
        minmax: str = "equal",
        fillcolor_missing_val: str = "#91FF9D",
        node_text_attributes: Union[List[str], Dict[str, str]] = "default",
        cmap: str = matplotlib.cm.RdBu_r,
    ) -> pydotplus.Dot:
        """Make a pydotplus.Dot from a NBNode tree.

        Args:
            tree (NBNode):
                The tree which should be plotted
            exported_dot_graph (str, optional):
                You can give your custom dot graph.
                Defaults to None, then it is exported internally by
                ``tree.export_dot()``.
            title (str, optional):
                Title for the plot. Creates an additional node holding the title.
                Defaults to None.
            fillcolor_node_attribute (str, optional):
                The (str) name of each node containing the numeric value how the node
                should be colored.
                Defaults to "counter".
            custom_min_max_dict (Dict[str, float], optional):
                You can give a custom dict with the min and max values for the
                fillcolor_node_attribute in the colorbar range.
                Takes precedence over minmax.

                Defaults to None, therefore is created by the minimum and maximum values
                from the fillcolor_node_attribute.
            minmax (str, optional):
                If custom_min_max_dict is None, the node filling colors reach their
                maximum/minimum color at the extremes of fillcolor_node_attribute::

                    "equal":
                        The colorbar is centered around 0, so the colorbar reaches from
                        -max(abs(minimum), abs(maximum))
                        to
                        +max(abs(minimum), abs(maximum)).
                    else:
                        The colorbar reaches from minimum to maximum.
                        Not necessarily symmetric.

                Defaults to "equal".
            fillcolor_missing_val (str, optional):
                Color for missing values.

                Defaults to "#91FF9D".
            node_text_attributes (Union[List[str], Dict[str, str]], optional):
                List or dict of attributes which should be displayed in the node:

                    - If "default", the node name is displayed.
                    - If a list, the list elements are used as keys for the node
                    attributes.
                    - If a dict, the dict values are used as keys for the node
                    attributes
                    and the dict values are used as format strings for the node
                    attributes.

                Defaults to "default".
            cmap (str, optional):
                The colormap which should be used for the fillcolor.

                Defaults to matplotlib.cm.RdBu_r.

        Returns:
            pydotplus.Dot: A graphviz Dot element, therefore a plot.
        """
        if node_text_attributes == "default":
            node_text_attributes = {"name": "{}"}

        if exported_dot_graph is None:
            exported_dot_graph = tree.export_dot()
        if tree.id_unique_dot_exporter is None:
            tree.set_DotExporter_ids()

        graph: pydotplus.Dot = pydotplus.graph_from_dot_data(exported_dot_graph)
        nodes = graph.get_node_list()

        # pydotplus seems to add an additional "node-like" element which is no node
        # but a new line which has just "\\n" inside. I dont know why.
        nodes = [node for node in nodes if node.get_name() != '"\\n"']

        if title is not None:
            # create a new node for text
            titlenode = pydotplus.Node("plottitle", label=title)
            titlenode.set_fillcolor("white")
            # add that node to the graph (No connections until now!)
            graph.add_node(titlenode)
            nodes = [
                node for node in graph.get_node_list() if node.get_name() != '"\\n"'
            ]
            # identify the first node
            node_zero = [
                node for node in nodes if node.get_name() == '"' + hex(id(tree)) + '"'
            ]
            if len(node_zero) != 0:
                # Then the node_zero could be found and the edge is created
                # Otherwise the edge is not created

                # add an edge between the plottitle and the first node
                # make it white
                # https://stackoverflow.com/questions/44274518/how-can-i-control-within-level-node-order-in-graphvizs-dot
                myedge = pydotplus.Edge(titlenode, node_zero[0])
                # add the edge to the graph
                graph.add_edge(myedge)

        # get the predicted (regression!) values from each node in the tree
        node_values = []
        for node in nodes:
            current_node_name = node.get_name()
            current_node_name = re.sub(r'"$', "", current_node_name)
            current_node_name = re.sub(r'^"', "", current_node_name)
            if current_node_name not in ("node", "edge", "plottitle"):
                current_matching_tree_node = anytree.search.find_by_attr(
                    node=tree, value=current_node_name, name="id_unique_dot_exporter"
                )
                if current_matching_tree_node is None:
                    raise ValueError("Node not found in tree!")
                node_values.append(
                    getattr(current_matching_tree_node, fillcolor_node_attribute)
                )

        if custom_min_max_dict is not None:
            vminmax = custom_min_max_dict
        else:
            all_values_concat = np.array(node_values)
            all_values_concat = all_values_concat[
                np.logical_not(np.isnan(all_values_concat))
            ]
            # the following command is the same, just shorthand
            # all_values_concat = all_values_concat[~np.isnan(all_values_concat)]
            if minmax == "equal":
                # then take the maximum of the absolute values and use it as positive
                # and negative colorbar
                all_values_concat_abs = np.abs(all_values_concat)
                vminmax = {
                    "min": -np.max(np.max(all_values_concat_abs), 0),
                    "max": np.max(np.max(all_values_concat_abs), 0),
                }
            else:
                vminmax = {
                    "min": np.min(np.min(all_values_concat), 0),
                    "max": np.max(np.max(all_values_concat), 0),
                }
        new_cmap = shifted_colormap(
            cmap=cmap,
            min_val=vminmax["min"],
            max_val=vminmax["max"],
            name="NoName",
        )
        norm = matplotlib.colors.Normalize(vmin=vminmax["min"], vmax=vminmax["max"])
        for node in nodes:
            current_node_name = node.get_name()
            current_node_name = re.sub(r'"$', "", current_node_name)
            current_node_name = re.sub(r'^"', "", current_node_name)
            if current_node_name not in ("node", "edge", "plottitle"):
                current_matching_tree_node = anytree.search.find_by_attr(
                    node=tree, value=current_node_name, name="id_unique_dot_exporter"
                )
                current_node_value = getattr(
                    current_matching_tree_node, fillcolor_node_attribute
                )
                if np.isnan(current_node_value):
                    node.set_fillcolor(fillcolor_missing_val)
                    node.set_shape("house")
                else:
                    node.set_fillcolor(
                        to_hex(new_cmap(norm(current_node_value)), keep_alpha=True)
                    )

                if isinstance(node_text_attributes, list):
                    node_value_attributes = [
                        str(value_name)
                        + ": "
                        + str(getattr(current_matching_tree_node, value_name))
                        for value_name in node_text_attributes
                    ]
                elif isinstance(node_text_attributes, dict):
                    node_value_attributes = [
                        str(value_name)
                        + ": "
                        + value_fmtstr.format(
                            getattr(current_matching_tree_node, value_name)
                        )
                        for value_name, value_fmtstr in node_text_attributes.items()
                    ]
                else:
                    node_value_attributes = node_text_attributes
                node.get_attributes()["label"] = "\n".join(node_value_attributes)
                # node.('\n'.join(node_value_attributes))

        return graph

    def insert_nodes(self, node_list: List[anytree.NodeMixin], copy_list: bool = False):
        """Insert nodes with the current node as parent.

        For every node in node_list, create a child node of self.

        Args:
            node_list:
                A list of nodes. All of these nodes get the parent set to the current
                node
            copy_list:
                If you reuse a list of nodes, e.g. twice, the nodes will not be assigned
                to both insertion nodes but RE-assigned to ONE of them.
                To omit this, copy the nodes first.

        """
        if copy_list:
            node_list = copy.deepcopy(node_list)
        for node_i, node_x in enumerate(node_list):
            if node_x.parent is not None:
                warnings.warn(
                    "The parent of the "
                    + str(node_i)
                    + ". node is not None, did you reuse the list "
                    + "without copy_list=True? The nodes would get RE-assigned, "
                    + "not additionally!"
                )
            node_x.parent = self

    def export_counts(
        predicted_celltree: "NBNode",
        only_leafnodes: bool = False,
        node_counts_dtype="int64",
    ) -> pd.DataFrame:
        """Export the counts of the predicted celltree to a pd.Dataframe.

        Rows are the samples, columns the node names `get_name_full()`

        Args:
            predicted_celltree (NBNode):
                The tree whose counts should be exported
            only_leafnodes (bool, optional):
                Should only leaf nodes (or all nodes) be counted/exported?
                Defaults to False.
            node_counts_dtype (str, optional):
                The dtype of the resulting counts. Defaults to "int64".
        Returns:
            pd.DataFrame:
                Rows are the samples, columns the node names `get_name_full()`.
        """
        predicted_celltree.count(use_ids=True)

        leaf_nodes_dict = {}
        all_nodes_dict = {}
        if predicted_celltree.data.shape[0] == 0:
            raise ValueError(
                "predicted_celltree.data is empty. Did you set it and make id_preds()?"
                + "\na = celltree_trunk.predict(cellmat)"
                + "\npredicted_celltree.data = pd.DataFrame()"
                + "\ncelltree_trunk.id_preds(a)"
            )

        for x in anytree.PostOrderIter(predicted_celltree):
            x: NBNode
            if "sample_name" not in predicted_celltree.data.columns:
                # Then I assume all cells in the node are from the same sample
                node_counts = x.data.shape[0]
                node_counts = pd.Series(node_counts, dtype=node_counts_dtype)
            else:
                node_counts = x.data["sample_name"].value_counts()
                node_counts.sort_index(inplace=True)
            if only_leafnodes:
                if x.is_leaf:
                    leaf_nodes_dict[x.get_name_full()] = node_counts
            else:
                all_nodes_dict[x.get_name_full()] = node_counts

        if only_leafnodes:
            counted_pops = leaf_nodes_dict
        else:
            counted_pops = all_nodes_dict
        counts_allsamples = pd.DataFrame().from_dict(counted_pops)
        counts_allsamples.fillna(0, inplace=True)
        counts_allsamples["Sample"] = counts_allsamples.index
        counts_allsamples.set_index("Sample", inplace=True)
        counts_allsamples = counts_allsamples.astype(node_counts_dtype)
        return counts_allsamples

    @staticmethod
    def edge_label_fun(decision_names, decision_values):
        """Function to label the edges of the tree."""
        if not isinstance(decision_names, list):
            decision_names = [decision_names]
            decision_values = [decision_values]
        generated_labels = []
        for name, value in zip(decision_names, decision_values):
            generated_labels.append("({}: {})".format(name, value))
        gathered_label = " & ".join(generated_labels)
        return 'label="{}"'.format(gathered_label)

    def both_iterator(
        self, other: "NBNode", strict: bool = False
    ) -> Tuple["NBNode", "NBNode"]:
        """Iterates over self and other simultaneously.

        Gives (yields) the same nodes until EITHER tree is at its end.

        Args:

            other (NBNode):
                Another NBNode object
            strict (bool, optional):
                If True, the trees must have the same nodes.
                Is only added in python 3.10 though.

        Yields:
            Another NBNode object
        """
        for node_self, node_other in zip(
            anytree.iterators.PreOrderIter(self),
            anytree.iterators.PreOrderIter(other),
            # https://stackoverflow.com/questions/32954486/zip-iterators-asserting-for-equal-length-in-python
            # strict=strict,  # is only added in python 3.10
        ):
            yield ((node_self, node_other))

    def copy_structure(self) -> "NBNode":
        """Copy only the structure of the tree.

        This does not copy the data, the ids or the counts.
        It copies additionally set attributes.

        Returns:
            NBNode: _description_
        """
        new_node = copy.deepcopy(self)
        new_node.reset_counts()
        new_node.reset_ids()
        new_node.data = None
        return new_node

    def eq_structure(self, other: "NBNode") -> bool:
        """Check if the structure of two trees is equal.

        It only checks node.name, node.decision_name and node.decision_value.
        It disregards the data, ids, counts or any other attribute.

        Args:
            other (NBNode): The other node to compare to

        Returns:
            bool: True if equal
        """
        if len(self) != len(other):
            return False

        for x, y in self.both_iterator(other):
            if not all(
                [
                    x.name == y.name,
                    x.decision_name == y.decision_name,
                    x.decision_value == y.decision_value,
                ]
            ):
                return False
        return True

    def astype_math_node_attribute(self, dtype, inplace=True) -> "NBNode":
        """Replaces all node.math_node_attribute with the given dtype.

        Args:
            dtype (_type_): The target type
            inplace (bool, optional):
                Replace inplace or not?.
                Defaults to True.

        Returns:
            NBNode: Returns self if inplace=True, else a copy of self
        """
        if not inplace:
            root = copy.deepcopy(self)
        else:
            root = self
        for node in anytree.iterators.PreOrderIter(root):
            setattr(
                node,
                self.math_node_attribute,
                (dtype)(getattr(node, self.math_node_attribute)),
            )
        return root

    def join(
        self,
        other: "NBNode",
        add_source_name: str = None,
        add_source_self: str = "self",
        add_source_other: str = "other",
        inplace: bool = True,
    ) -> "NBNode":
        """Join two NBNodes.

        The NBNodes must match in structure.

            1. The nodes are added --> math_node_attribute of both NBNodes
            2. The data of other is added to the data of self
            3. The ids of other are added to the ids of self according to the new _data

        Args:
            other (NBNode): The other NBNode to join
            add_source_name (str, optional):
                If True, when joining the data an additional column is created
                with the name `add_source_name` which then contains either
                ``add_source_self`` or ``add_source_other`` depending on the source
                of the data.

                Defaults to None, so the column is not created.
            add_source_self (str, optional):
                The value to use for the column ``add_source_name`` when the data
                is joined.

                Defaults to "self".
            add_source_other (str, optional):
                The value to use for the column ``add_source_name`` when the data
                is joined

                Defaults to "other".
            inplace (bool, optional):
                If True, the NBNode is modified inplace.
        Returns:
            NBNode: A single NBNode with the data of both NBNodes
        """
        if not inplace:
            self = copy.deepcopy(self)

        # Add math_node_attribute of both NBNodes
        self = self + other

        self_total_predicted_nodes_n = len(self.root.ids)
        other_data = other.root._data
        self_data = self.root._data

        if add_source_name is not None:
            other_data = other_data.assign(**{add_source_name: add_source_other})
            if add_source_name not in self.data.columns:
                self_data = self_data.assign(**{add_source_name: add_source_self})

        self.root._data = pd.concat([self_data, other_data], ignore_index=True)

        for node_self, node_other in self.both_iterator(other):
            node_self: NBNode
            node_other: NBNode
            other_new_ids = [
                old_id + self_total_predicted_nodes_n for old_id in node_other.ids
            ]
            node_self.ids += other_new_ids
        return self

    def __len__(self) -> int:
        """Get the number of nodes in the tree.

        Returns:
            int: The number of nodes in the tree.
        """
        length = 0
        for node in anytree.iterators.PreOrderIter(self):
            length += 1
        return length

    def pretty_print(
        self, print_attributes: List[str] = "__default__", round_ndigits: int = None
    ):
        """Print the tree in a pretty way.

        Args:
            print_attributes (List[str], optional):
                The attributes of each NBnode which should be printed.
                Defaults to "__default__", then only the `counter` attribute is shown.

            round_ndigits (int, optional):
                If not None, the values of the attributes are rounded to the given
                number of digits.

                Defaults to None.
        """
        if print_attributes == "__default__":
            print_attributes = ["counter"]
        if print_attributes == "__long__":
            print_attributes = self._long_print_attributes

        for pre, fill, node in anytree.RenderTree(self):
            node_attrs = []
            if print_attributes is not None:
                attr_values = [getattr(node, x) for x in print_attributes]
                if round_ndigits is not None:
                    attr_values = [round(x, ndigits=round_ndigits) for x in attr_values]
                attr_names = [x for x in print_attributes]
                node_attrs = [
                    f"{name}:{value}" for name, value in zip(attr_names, attr_values)
                ]

            joined_attr = ", ".join(str(x) for x in node_attrs)
            print(f"{pre}{node.name} ({joined_attr})")

    def __both_nodeattr_fun(
        self,
        other: "NBNode",
        fun=lambda x, y: x + y,
        strict: bool = True,
        inplace: bool = None,
        type_force_fun_names=(
            "__truediv__",
            "__divmod__",
        ),
    ) -> "NBNode":
        """Apply a function to the "same" node of two NBNodes.

        Traverses both trees simultaneously and applies
        ``fun(node_1.math_node_attribute, node_2.math_node_attribute)``.

        Args:
            other (NBNode): The other NBNode
            fun (_type_, optional):
                The function to apply to the math_node_attribute of all nodes from
                both NBNodes.

                Defaults to ``lambda x, y: x + y``.
            strict (bool, optional):
                See `NBNode.both_iterator()`.
                Defaults to True.
            inplace (bool, optional):
                If True, the NBNode is modified inplace.
                Defaults to None.
            type_force_fun_names (tuple, optional):
                Some functions need to be forced to be applied to the same type of the
                math_node_attribute. Use with care!
                Better, use ``NBNode.astype_math_node_attribute()`` before.

                Defaults to ( "__truediv__", "__divmod__", ).

        Returns:
            NBNode:
                A new NBNode with the result of the function applied to the
                math_node_attribute of both NBNodes.
                The result is directly saved in the math_node_attribute of the new
                NBNode.
        """
        self_tmp = self
        if inplace is None:
            inplace = self.math_inplace
        if not inplace:
            self_tmp = copy.deepcopy(self)

        if isinstance(fun, str):
            # Then fun should be `inspect.stack()[0][3]` and be something like "__add__"
            # Then the type of the current node_attr is determined (e.g. float)
            attr_type = type(getattr(self, self_tmp.math_node_attribute))
            # and fun() will become float.__add__()
            fun = getattr(attr_type, fun)
        if isinstance(other, NBNode):
            for node_self, node_other in self_tmp.both_iterator(other, strict=strict):
                a = getattr(node_self, self_tmp.math_node_attribute)
                b = getattr(node_other, self_tmp.math_node_attribute)
                if fun.__name__ == type_force_fun_names:
                    a = type(b)(a) if not isinstance(a, type(b)) else a
                setattr(
                    node_self,
                    self_tmp.math_node_attribute,
                    fun(a, b),
                )
        else:
            for node_self in anytree.iterators.PreOrderIter(self_tmp):
                a = getattr(node_self, self_tmp.math_node_attribute)
                b = other
                if fun.__name__ in type_force_fun_names:
                    a = type(b)(a) if not isinstance(a, type(b)) else a
                setattr(
                    node_self,
                    self_tmp.math_node_attribute,
                    fun(a, b),
                )
        return self_tmp

    def __eq__(self, other) -> bool:
        """Check if two NBNodes are equal.

        Args:
            other (NBnode): Other NBnode to compare with.

        Returns:
            bool:
                True if the two NBNodes are equal regarding structure and the attributes
                name, decision_name, decision_value, counter, ids.
                False otherwise.
        """
        if not isinstance(other, NBNode):
            return False
        if not self.eq_structure(other):
            return False
        for x, y in self.both_iterator(other):
            if not all(
                [
                    x.name == y.name,
                    x.decision_name == y.decision_name,
                    x.decision_value == y.decision_value,
                    x.counter == y.counter,
                    x.ids == y.ids,
                ]
            ):
                return False
        return True

    def __hash__(self):
        return hash(self.__repr__())

    def __add__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __sub__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __mul__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __floordiv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __div__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __truediv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __mod__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __divmod__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __lshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __and__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __or__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __xor__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __radd__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rsub__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rmul__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rfloordiv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rdiv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rtruediv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rmod__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rdivmod__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rlshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rrshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rand__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __ror__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __rxor__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __iadd__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __isub__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __imul__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __ifloordiv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __idiv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __itruediv__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __imod__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __ilshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __irshift__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __iand__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __ior__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __ixor__(self, other):
        return self.__both_nodeattr_fun(other=other, fun=inspect.stack()[0][3])

    def __repr__(self, include_attrs="default"):
        if include_attrs == "default":
            include_attrs = self._long_print_attributes
        exclude_attrs = [key for key in self.__dict__ if key not in include_attrs]

        args = [
            "%r" % self.separator.join([""] + [str(node.name) for node in self.path])
        ]
        return anytree.node.util._repr(self, args=args, nameblacklist=exclude_attrs)

    def get_name_full(self) -> str:
        """Get the full name of the node (including the root node "/")

        Returns:
            str: Full name of the node.
        """
        full_name = ""
        current_node = self
        while not current_node.is_root:
            full_name = current_node.name + "/" + full_name
            current_node = current_node.parent
        full_name = "/" + current_node.name + "/" + full_name

        return full_name[:-1]  # remove the last "/"

    def __getitem__(self, nbnode_full_name: str) -> "NBNode":
        """Get a node by its full name (including the root node "/")

        Args:
            nbnode_full_name (str): Full name of the node.

        Raises:
            ValueError: If the selector is not a string with a full name from some node

        Returns:
            NBNode: The node with the given full name.
        """
        if nbnode_full_name == 0:
            # This happens if printing predicted nodes (which are a pandas.DataFrame)
            #   predicted_nodes = celltree_decision.predict(tmp_data)
            #   print(predicted_nodes)
            return self
        if not isinstance(nbnode_full_name, str):
            raise ValueError(
                "The selector must be a string with a full name from some node "
                + "(created by `node.get_name_full()`)"
            )
        return anytree.find(self, lambda node: node.get_name_full() == nbnode_full_name)
