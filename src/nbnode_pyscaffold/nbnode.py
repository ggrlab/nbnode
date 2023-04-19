import inspect
from typing import Tuple, Union, List, Dict, Optional, Any

import anytree
from anytree.exporter import UniqueDotExporter
import copy
import warnings

import pandas as pd
import datatable
from collections import Counter

import pydotplus
import re
import matplotlib

matplotlib.use("AGG")
from matplotlib.colors import to_hex
import numpy as np

from nbnode_pyscaffold.utils.shifted_colormap import shifted_colormap


class NBNode(anytree.Node):
    """
    Non-binary node class.

    Args:

        anytree (anytree.Node) :
            The base anytree class.

    """

    def __init__(
        self,
        name,
        parent=None,
        decision_value=None,
        decision_name=None,
        decision_cutoff=None,
        **kwargs,
    ):
        super(NBNode, self).__init__(name=name, parent=parent, **kwargs)
        self.parent = parent
        self.decision_name = decision_name
        self.decision_value = decision_value
        self.decision_cutoff = decision_cutoff  # Not necessary in ternary prediction
        self.counter = 0
        self.ids = []
        self._data = None
        self.id_unique_dot_exporter = None
        # Set math_node_attribute to whatever you want to add when using (usual) mathematics.
        self.math_node_attribute = "counter"
        self.math_inplace = False
        self._long_print_attributes = ["counter", "decision_name", "decision_value"]

    def prediction_str(self, nodename: str, split: str = "/"):
        if isinstance(nodename, str):
            split_names = nodename.split(split)
            # if first element is "", then because that was the root.
            if split_names[0] == "":
                split_names = split_names[1:]
        else:
            split_names = nodename

        if split_names[0] != self.name:
            raise ValueError(
                "self.name != split_names[0]. Did you start at (=supply) the right node?"
            )
        if len(self.children) != 0 and len(split_names) > 1:
            for child in self.children:
                if child.name == split_names[1]:
                    # "Remove" [1:] the first element (the current node's name)
                    return child.prediction_str(split_names[1:])
        else:
            return self

    @staticmethod
    def do_cutoff(value, cutoff):
        # ">= --> 1" from https://en.wikipedia.org/wiki/Decision_tree_learning
        if value >= cutoff:
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
        """
        Predicts the endnode (leaf) of the tree given the values.

        Args:
            values:
            names:
            allow_unfitting_data:
            False:  If the data you gave was not possible to fit in the tree raises a ValueError.
            True:   If the data you gave was not possible to fit in the tree returns None.
        Returns:
            Either a single WNode instance (the leaf node) or if multiple leaf nodes fit, all of them as a list.
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
        #     raise ValueError('Could not find a fitting endnode for the data you gave.')
        else:
            if len(leaf_nodes_list) == 1:
                leaf_nodes_list = leaf_nodes_list[0]
            return leaf_nodes_list

    def predict(
        self,
        values: Union[List, Dict, pd.DataFrame],
        names: list = None,
        allow_unfitting_data: bool = False,
        allow_part_predictions: bool = False,
    ) -> Union["NBNode", List["NBNode"], pd.Series]:
        """
        See single_prediction, but you can put in dataframes or ndarrays instead of only dict + value/key paired lists.

        Returns a list of nodes.
        """
        if isinstance(values, datatable.Frame):
            values = values.to_pandas()

        if isinstance(values, np.ndarray):
            if names is None:
                raise ValueError(
                    "You supplied a numpy array but no names. Names are necessary with np.ndarray"
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
        """

        Args:
            node_list:
                The [usually predicted] nodes. Usual workflow would be:
                    1. Predict n samples
                    2. Get a list of n nodes from these predictions

                But you can insert here any node (inside the tree) you want.
            reset_counts:
                Should all .counter be set to 0?
            use_ids:
                If use_ids==True, do not use node_list to count but just access the length of the node.ids

        Returns:

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
        self.ids = []

        for child in self.children:
            if child is not None:
                child.reset_ids()

    def id_preds(self, node_list: List["NBNode"], reset_ids: bool = True):
        """Predict node ids

        Given a list of nodes, enumerate through them and assign this (`enumerate(node_list)`)
        number to self.ids.

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
        if self.ids == []:
            warnings.warn(
                "self.ids was an empty list, subset an empty dataframe. Did you call celltree.id_preds(predicted_nodes)? Can also be a node with no cells."
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
        """
        Apply the given function to the `.data` property of each node.
        Args:
            fun:
            result_attribute_name:
                If result_attribute_name is given, the return value of `fun(node.data)` is set to the node's `result_attribute_name`-attribute
            iterator:
                How to iterate over the nodes.
        Returns:

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

    def export_dot(self, unique_dot_exporter_kwargs: Dict = "default"):
        if unique_dot_exporter_kwargs == "default":
            unique_dot_exporter_kwargs = {
                "options": ['node [shape=box, style="filled", color="black"];'],
                "nodeattrfunc": lambda node: 'label="{}", fillcolor="white"'.format(
                    node.name
                ),
            }
        dot_data = UniqueDotExporter(self, **unique_dot_exporter_kwargs)
        dotdata_str = "\n".join([x for x in dot_data])
        return dotdata_str

    def set_uniquedotexporter_ids(self):
        for node in anytree.iterators.PreOrderIter(self):
            node.id_unique_dot_exporter = hex(
                id(node)
            )  # That is how UniqueDotExporter gets his values

    @staticmethod
    def graph_from_dot(
        tree: "NBNode",
        exported_dot_graph: str = None,
        title: str = None,
        fillcolor_node_attribute: str = "height",
        custom_min_max_dict: Dict[str, float] = None,
        minmax: str = "equal",
        fillcolor_missing_val: str = "#91FF9D",
        node_text_attributes: Union[List[str], Dict[str, str]] = "default",
    ) -> pydotplus.Dot:
        if node_text_attributes == "default":
            node_text_attributes = {"name": "{}"}

        if exported_dot_graph is None:
            exported_dot_graph = tree.export_dot()
        if tree.id_unique_dot_exporter is None:
            tree.set_uniquedotexporter_ids()

        graph: pydotplus.Dot = pydotplus.graph_from_dot_data(exported_dot_graph)
        nodes = graph.get_node_list()

        # pydotplus seems to add an additional "node-like" element which is no node but a
        # new line which has just "\\n" inside. I dont know why.
        nodes = [node for node in nodes if node.get_name() != '"\\n"']

        if title is not None:
            # create a new node for text
            titlenode = pydotplus.Node("plottitle", label=title)
            titlenode.set_fillcolor("white")
            # add that node to the graph (No connections until now!)
            graph.add_node(titlenode)
            # identify the first node
            node_zero = [node for node in nodes if node.get_name() == "0"]
            # add an edge between the plottitle and the first node
            # make it white https://stackoverflow.com/questions/44274518/how-can-i-control-within-level-node-order-in-graphvizs-dot
            myedge = pydotplus.Edge(titlenode, node_zero[0])
            # add the edge to the graph
            graph.add_edge(myedge)

        # get the predicted (regression!) values from each node in the tree
        node_values = []
        for node in nodes:
            current_node_name = node.get_name()
            current_node_name = re.sub(r'"$', "", current_node_name)
            current_node_name = re.sub(r'^"', "", current_node_name)
            if current_node_name not in ("node", "edge"):
                current_matching_tree_node = anytree.search.find_by_attr(
                    node=tree, value=current_node_name, name="id_unique_dot_exporter"
                )
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
            # all_values_concat = all_values_concat[~np.isnan(all_values_concat)]  # same, just shorthand
            if minmax == "equal":
                # then take the maximum of the absolute values and use it as positive and negative colorbar
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
            cmap=matplotlib.cm.RdBu_r,
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
        """

        Args:
            node_list:
                A list of nodes. All of these nodes get the parent set to the current node
            copy_list:
                If you reuse a list of nodes, e.g. twice, the nodes will not be assigned to both insertion nodes but
                RE-assigned to ONE of them. To omit this, copy the nodes first.

        Returns:

        """
        if copy_list:
            node_list = copy.deepcopy(node_list)
        for node_i, node_x in enumerate(node_list):
            if node_x.parent is not None:
                warnings.warn(
                    "The parent of the "
                    + str(node_i)
                    + ". node is not None, did you reuse the list "
                    + "without copy_list=True? The nodes would get RE-assigned, not additionally!"
                )
            node_x.parent = self

    def export_counts(predicted_celltree: "NBNode", only_leafnodes: bool = False):
        predicted_celltree.count(use_ids=True)

        # leaf_nodes = [x for x in anytree.PostOrderIter(celltree_decision) if x.is_leaf]
        leaf_nodes_dict = {}
        all_nodes_dict = {}
        for x in anytree.PostOrderIter(predicted_celltree):
            x: NBNode
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
        return counts_allsamples

    @staticmethod
    def edge_label_fun(decision_names, decision_values):
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
        """Iterates over self and other simultaneously

        Yields the same nodes until EITHER tree is at its end

        Args:
            other (NBNode): Another NBNode object
        """
        for node_self, node_other in zip(
            anytree.iterators.PreOrderIter(self),
            anytree.iterators.PreOrderIter(other),
            # https://stackoverflow.com/questions/32954486/zip-iterators-asserting-for-equal-length-in-python
            # strict=strict,  # is only added in python 3.10
        ):
            yield ((node_self, node_other))

    def copy_structure(self) -> "NBNode":
        new_node = copy.deepcopy(self)
        new_node.reset_counts()
        new_node.reset_ids()
        new_node.data = None
        return new_node

    def eq_structure(self, other: "NBNode"):
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

    def join(
        self,
        other: "NBNode",
        add_source_name: str = None,
        add_source_self: str = "self",
        add_source_other: str = "other",
    ) -> "NBNode":
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

    def __len__(self):
        length = 0
        for node in anytree.iterators.PreOrderIter(self):
            length += 1
        return length

    def pretty_print(
        self, print_attributes: List[str] = "__default__", round_ndigits: int = None
    ):
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
        self, other, fun=lambda x, y: x + y, strict: bool = True, inplace: bool = None
    ) -> "NBNode":
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
                setattr(
                    node_self,
                    self_tmp.math_node_attribute,
                    fun(
                        getattr(node_self, self_tmp.math_node_attribute),
                        getattr(node_other, self_tmp.math_node_attribute),
                    ),
                )
        else:
            for node_self in anytree.iterators.PreOrderIter(self_tmp):
                setattr(
                    node_self,
                    self_tmp.math_node_attribute,
                    fun(
                        getattr(node_self, self_tmp.math_node_attribute),
                        other,
                    ),
                )
        return self_tmp

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

    def get_name_full(self):
        full_name = ""
        current_node = self
        while not current_node.is_root:
            full_name = current_node.name + "/" + full_name
            current_node = current_node.parent
        full_name = "/" + current_node.name + "/" + full_name

        return full_name[:-1]  # remove the last "/"

    def __getitem__(self, nbnode_full_name: str) -> "NBNode":
        if nbnode_full_name == 0:
            # This happens if printing predicted nodes (which are a pandas.DataFrame)
            #   predicted_nodes = celltree_decision.predict(tmp_data)
            #   print(predicted_nodes)
            return self
        if not isinstance(nbnode_full_name, str):
            raise ValueError(
                "The selector must be a string with a full name from some node (created by `node.get_name_full()`)"
            )
        return anytree.find(self, lambda node: node.get_name_full() == nbnode_full_name)