

class Version:
    """
    A version number object
    This makes it easier to sort and compare metadata versions.
    """

    def __init__(self, version=None, minor=None, patch=None):
        """
        Construct a version object.
        Can parse a lot of different parameter forms,
        Version()           : Default version 0.0.0
        Version(other)      : Copy constructor
        Version(1,2,3)      : Construct using 3 integers, 1.2.3
        Version("1.2.3")    : Parse the string, produces 1.2.3
        Version((1,2,3))    : Parse the tuple into 1.2.3
        Version([1,2,3])    : Parse the list into 1.2.3

        :param version:
        :param minor:
        :param patch:
        """
        if isinstance(version, Version):
            self._major = version._major
            self._minor = version._minor
            self._patch = version._patch
        elif isinstance(version, int) and isinstance(minor, int) and isinstance(patch, int):
            self._major = version
            self._minor = minor
            self._patch = patch
        elif isinstance(version, str):
            parts = version.split('.')
            self._major = int(parts[0])
            self._minor = int(parts[1]) if len(parts) >= 2 else 0
            self._patch = int(parts[2]) if len(parts) >= 3 else 0
        elif (isinstance(version, tuple) or isinstance(version, list)) and len(version) >= 3:
            self._major = int(version[0])
            self._minor = int(version[1])
            self._patch = int(version[2])
        else:
            self._major = 0
            self._minor = 0
            self._patch = 0

    def __gt__(self, other):
        """
        Implementation for self > other.
        All the other rich comparisons are defined in terms of this and eq
        :param other:
        :return:
        """
        if not isinstance(other, Version):
            other = Version(other)
        return self.major > other.major or (
            self.major == other.major and (
                self.minor > other.minor or (
                    self.minor == other.minor and self.patch > other.patch
                )
            )
        )

    def __lt__(self, other):
        return not (self > other or self == other)

    def __ge__(self, other):
        return self == other or self > other

    def __le__(self, other):
        return not self > other

    def __eq__(self, other):
        if not isinstance(other, Version):
            other = Version(other)
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return '{0}.{1}.{2}'.format(self.major, self.minor, self.patch)

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def patch(self):
        return self._patch
