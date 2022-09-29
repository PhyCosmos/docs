# 1. Namespace in Python

- 같은 이름 사용으로 인한 충돌을 막기 위해 
- scope resolution으로 구분되는 
- `{name: obj}` dict 타입으로 정의된 것
- Built-in Namespace > Global Namespace > Enclosing Namespace > Local Namespace
- Enclosing Namespace는 function이 다른 function을 포함하는 경우에 생긴다.|


```python
# 1. Built-in Namespace
dir(__builtins__)[97:100]
```
    ['dict', 'dir', 'display']



```python
# 2. Global Namespace
x = 'global'
def f():
    # x = 'enclosign'
    def g():
        # x = 'local'
        print(x)
    g()

print(globals()['__name__'])
f() in globals().values()
```

    __main__
    global

    True


```python
# 3. Enclosing Namespace
x = 'global'
def f():
    x = 'enclosign'
    def g():
        # x = 'local'
        print(x)
    g()
    
f()
```

    enclosign



```python
# 4. local Namespace
x = 'global'
def f():
    x = 'enclosign'
    def g():
        x = 'local'
        print(x)
    g()
    
f()
```

    local

