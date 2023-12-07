import numpy as np

DEBUG = False


def is_jitlass(func):
    return str(type(func)) == "<class 'numba.experimental.jitclass.base.JitClassType'>"


class RemoveNumbaSettings:
    def __init__(self, allowed_packes=tuple()):
        self.allowed_packages = tuple(allowed_packes) + ("__main__",)
        self.search_deep = True
        self.depth = 0
        self.max_depth = 1000

    def is_allowed_module_string(self, module_string):
        if module_string.startswith("numba.experimental.jitclass.base"):
            return True
        if module_string == "":
            return True
        return any(
            package_str in module_string for package_str in self.allowed_packages
        )


DefaultSettings = RemoveNumbaSettings()


def track_namespace(namespace, seen_namespaces):
    if namespace.id() in seen_namespaces:
        return True
    else:
        seen_namespaces[namespace.id()] = namespace
        return False


def is_right_module(obj, settings):
    if str(type(obj)) == "<class 'module'>" and not settings.is_allowed_module_string(
        obj.__package__
    ):
        return False
    if repr(obj).startswith("<class 'numba.experimental.jitclass.base."):
        return True
    if not settings.is_allowed_module_string(obj.__module__):
        # print(obj)
        # print(obj, obj.__module__)
        return False
    return True


class DictWrapper:
    def __init__(self, d, parents, name=None):
        self.d = d
        self.parents = parents
        if name is None:
            self._name = self.d["__name__"]
        else:
            self._name = name

    def id(self):
        return id(self.d)

    def set(self, key, value):
        self.d[key] = value

    def get(self, key):
        return self.d[key]

    def items(self):
        return self.d.items()

    def __eq__(self, other):
        return self.d is other.d

    def __repr__(self):
        return f"DictWrapper({self.parents},{filter_globals(self.d)})"

    def name(self):
        return self._name


class ModuleWrapper:
    def __init__(self, module, parents):
        self.module = module
        self.parents = parents

    def id(self):
        return id(self.module)

    def set(self, key, value):
        setattr(self.module, key, value)

    def get(self, key):
        return getattr(self.module, key)

    def items(self):
        return {key: getattr(self.module, key) for key in dir(self.module)}

    def __repr__(self):
        return f"ModuleWrapper({self.parents},{filter_globals(self.items())})"

    def name(self):
        return repr(self.module)


class ClassWrapper:
    def __init__(self, cls, parents):
        self.d = cls.__dict__
        self.parents = parents
        self._name = str(type(cls)).split("'")[1]

    def id(self):
        return id(self.d)

    def set(self, key, value):
        self.d[key] = value

    def get(self, key):
        return self.d[key]

    def items(self):
        return self.d.items()

    def __repr__(self):
        return f"ClassWrapper({self.parents},{filter_globals(dict(self.items()))})"

    def name(self):
        return self._name


class TypeWrapper:
    def __init__(self, cls, parents):
        self.cls = cls
        self.parents = parents
        self._name = str(cls).split("'")[1]

    def id(self):
        return id(self.cls)

    def set(self, key, value):
        setattr(self.cls, key, value)

    def get(self, key):
        return getattr(self.cls, key)

    def items(self):
        return {key: getattr(self.cls, key) for key in dir(self.cls)}.items()

    def __repr__(self):
        return f"TypeWrapper({self.parents},{filter_globals(dict(self.items()))})"

    def name(self):
        return self._name


def collect_unseen_namespaces(obj, seen_namespaces, settings, parents=tuple()):
    def my_print(*args):
        print("  " * settings.depth, *args)

    # my_print(obj, type(obj))
    # need to do modules firs because they have no __module__
    if str(type(obj)) == "<class 'module'>" and not settings.is_allowed_module_string(
        obj.__package__
    ):
        return
    # my_print(" ",obj, obj.__module__)
    # my_print(" ",obj.__init__.__globals__)
    if not settings.is_allowed_module_string(obj.__module__):
        # print("MODULE", obj.__module__)
        return
    # my_print(dir(obj))
    if hasattr(obj, "py_func"):  # is @njit function
        namespace = DictWrapper(obj.py_func.__globals__, parents)
        if track_namespace(namespace, seen_namespaces):
            return
    elif isinstance(obj, type(is_right_module)):
        # my_print("MODULE", obj.__module__)
        namespace = DictWrapper(obj.__globals__, parents)
        if track_namespace(namespace, seen_namespaces):
            return
    elif is_jitlass(obj):
        # print("JITCLASS")
        namespace = DictWrapper(obj.class_type.jit_methods, parents, name=str(obj))
        # print(obj.class_type.jit_methods)
        if track_namespace(namespace, seen_namespaces):
            return
    elif (
        str(type(obj)) == "<class 'module'>"
        and obj.__package__ in settings.allowed_packges
    ):
        namespace = ModuleWrapper(obj, parents)
        if track_namespace(ModuleWrapper, seen_namespaces):
            return
    elif str(type(obj)) == "<class 'type'>":
        namespace = TypeWrapper(obj, parents)
    elif str(type(obj)).startswith("<class '"):
        namespace = ClassWrapper(obj, parents)
    else:
        raise NotImplementedError(type(obj))
    # print("HERE", namespace)
    for key, new_obj in namespace.items():
        if key in (
            "__loader__",
            "__builtins__",
            "__dict__",
            "__weakref__",
            "__delattr__",
            "__dir__",
            "__new__",
            "@py_builtins",
        ):
            continue
        # print(key, type(new_obj))
        if str(type(new_obj)) in (
            "<class 'method_descriptor'>",
            "<class 'wrapper_descriptor'>",
            "<class 'builtin_function_or_method'>",
        ):
            continue
        if isinstance(new_obj, (list, dict, np.ndarray, tuple)):
            continue
        if isinstance(new_obj, (int, float, str)):
            continue
        if new_obj is None:
            continue
        if str(new_obj).startswith("ModuleSpec("):
            continue
        # print("BB", key, type(new_obj))
        if not is_right_module(new_obj, settings):
            continue
        # if key in ("_smawk_iter",):
        #    print(key, type(new_obj), parents)
        # my_print("into", key, type(new_obj))
        settings.depth += 1
        collect_unseen_namespaces(
            new_obj,
            seen_namespaces,
            settings,
            parents + (namespace.name() + "." + key,),
        )
        settings.depth -= 1


class UndoNjitFunction:
    def __init__(self, name):
        self.name = name
        self.njit_function = None

    def remove_numba(self, namespace):
        self.njit_function = namespace.get(self.name)
        namespace.set(self.name, self.njit_function.py_func)

    def add_numba(self, namespace):
        namespace.set(self.name, self.njit_function)

    def __repr__(self):
        return f"UndoNjitFunction({self.name})"

    def __eq__(self, other):
        if not isinstance(other, UndoNjitFunction):
            return False
        return self.name == other.name


class UndoJitclassFunction:
    def __init__(self, name):
        self.name = name
        self.njit_class = None
        self.python_class = None

    def remove_numba(self, namespace):
        self.njit_class = namespace.get(self.name)
        if self.njit_class is None:
            raise ValueError()
        # print(self.njit_class.class_type.jit_methods)
        new_methods = {
            key: getattr(value, "py_func", value)
            for key, value in self.njit_class.class_type.jit_methods.items()
        }

        self.python_class = type(self.njit_class.class_type.class_name, (), new_methods)
        namespace.set(self.name, self.python_class)

    def add_numba(self, namespace):
        namespace.set(self.name, self.njit_class)

    def __eq__(self, other):
        if not isinstance(other, UndoJitclassFunction):
            return False
        return self.name == other.name

    def __repr__(self):
        if self.njit_class is None:
            return f"UndoJitclassFunction({self.name})"
        return f"UndoJitclassFunction({self.njit_class.class_type.class_name})"


def is_trivial_object(key, obj):
    if key in ("__loader__", "__builtins__", "__dict__", "__weakref__"):
        return True
    if isinstance(obj, (list, dict, np.ndarray, tuple)):
        return True
    if isinstance(obj, (int, float, str)):
        return True
    if obj is None:
        return True
    return False


def remove_numba_from_namespaces(namespaces):
    list_of_undos = []
    for namespace in namespaces.values():
        undo_list = []
        list_of_undos.append(undo_list)
        for key, obj in namespace.items():
            if is_trivial_object(key, obj):
                continue
            if hasattr(obj, "py_func"):
                # print()
                # print(filter_globals(namespace))
                # print(key, obj)
                undo_list.append(UndoNjitFunction(key))
            elif is_jitlass(obj):
                undo_list.append(UndoJitclassFunction(key))
            else:
                pass
                # print(obj, type(obj))
        for changer in undo_list:
            # print(changer)
            # print(namespace)
            changer.remove_numba(namespace)
            # print()
            # print(changer)
            # print(namespace)
    return list_of_undos


def filter_globals(d):
    out = {}
    for key, value in d.items():
        if key.endswith("Error"):
            continue
        if key in (
            "quit",
            "copyright",
            "np",
            "__spec__",
            "__file__",
            "__loader__",
            "__cached__",
        ):
            continue
        if key in ("__builtins__", "unittest"):
            continue
        out[key] = value
    return out


def remove_numba_from_class(cls, settings=None, allowed_packages=None):
    if settings is None:
        if allowed_packages is None:
            raise ValueError()
        # print("CLS MODULE", cls.__module__)
        settings = RemoveNumbaSettings(
            allowed_packes=tuple(allowed_packages) + (cls.__module__,)
        )
    namespaces = {}
    collect_unseen_namespaces(cls, namespaces, settings)
    # print(len(namespaces))
    list_of_undos = remove_numba_from_namespaces(namespaces)

    namespaces2 = {}
    collect_unseen_namespaces(cls, namespaces2, settings)
    list_of_undos2 = remove_numba_from_namespaces(namespaces2)

    if DEBUG:
        print("-----------------")
        print([namespace.name() for namespace in namespaces.values()])
        print("-----------------")
        # print(len(namespaces))

        for (_, namespace), l in zip(namespaces2.items(), list_of_undos2):
            if not len(l) == 0:
                # print()
                # print("?????")
                # print(namespace)
                # print(l)
                assert False

    return namespaces, list_of_undos


def restore_to_class(stuff):
    namespaces, list_of_undos = stuff
    for (_, namespace), undos in zip(namespaces.items(), list_of_undos):
        for undo in undos:
            # print(undo)
            undo.add_numba(namespace)


def namespaces_equal(namespaces1, namespaces2):
    return list(namespaces1.keys()) == list(namespaces2.keys()) and all(
        [ns1 == ns2 for ns1, ns2 in zip(namespaces1.values(), namespaces2.values())]
    )


def undos_equal(undos1, undos2):
    for l1, l2 in zip(undos1, undos2):
        if not all([u1 == u2 for u1, u2 in zip(l1, l2)]):
            return False
    return True


def cleanups_equal(cleanup1, cleanup2):
    """Checks whether two cleanups are equal"""
    namespaces1, undos1 = cleanup1
    namespaces2, undos2 = cleanup2

    return namespaces_equal(namespaces1, namespaces2) and undos_equal(undos1, undos2)
