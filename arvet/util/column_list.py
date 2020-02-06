import typing


class ColumnList:
    """
    A helper for storing data column names.
    Is basically a mapping between column names, and callables for retrieving the value of that column from an object.
    Is immutable, define this on a class at construction time.
    The key functionality is to be able to delegate to parent mapping, so that if a mapping doesn't have a value
    for a column, it can use the parent one.
    This allows me to override and extend column lists in subclasses
    """

    def __init__(self, *args: 'ColumnList', **kwargs: typing.Union[None, typing.Callable[[typing.Any], typing.Any]]):
        self._parents = args
        self._columns = kwargs

    def __contains__(self, item: str) -> bool:
        return item in self._columns or any(item in parent for parent in self._parents)

    def keys(self) -> typing.Iterator[str]:
        """
        Loop over the column names
        :return:
        """
        for parent in self._parents:
            yield from parent.keys()
        yield from self._columns.keys()

    def get_value(self, model: typing.Any, column: str) -> typing.Any:
        if column in self._columns:
            if self._columns[column] is None:
                return float('nan')     # Return NaN, which pandas uses as value omitted.
            return self._columns[column](model)
        for parent in self._parents:
            if column in parent:
                return parent.get_value(model, column)
        raise KeyError("Could not find column '{0}'".format(column))
