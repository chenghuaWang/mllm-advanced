# Utils

## Dbg

Stop using `std::cout`, `printf`, or `MLLM_INFO(...)` for debugging purposes! Instead, opt for `Dbg()`, which is a more efficient and reliable choice.

Usage:

e.g.:

```cpp
// in file foo.cpp
#include "mllm/Utils/Dbg.hpp"

int foo() {
    Dbg();
}
```

It will output:

```text
dbg| foo.cpp:4, in int foo()
```

e.g.:


```cpp
// in file foo.cpp
#include "mllm/Utils/Dbg.hpp"

int foo(int i=6) {
    Dbg(i);
}
```

It will output:

```text
dbg| foo.cpp:4, i:6
```
