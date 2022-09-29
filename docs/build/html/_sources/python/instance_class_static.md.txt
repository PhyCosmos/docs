# 1. Methods in classes

class 안에 있는 method 는 오브젝트 `cls`와 인스턴스 `self`에 대한 access 방식에 따라 3 가지로 나눌 수 있다.  

<ul>
<li> 클래스 디자인의 의도를 강제하려는 목적과 </li>
<li> 그러한 의도를 쉽게 알아보게 하여 맞춤관리에 도움을 주고자 <s>할 뿐 엄격하지는 않다</s>. </li>
</ul>  

|메서드|접근방식|호출형식|
|:--|:--|:--|
|instance method|`self`|`obj = ClassName(); obj.instance_method()`|
|class method|`cls`| `ClassName.class_method()`|
|static method|없음 | `ClassName.static_mehtod()`|

- 1) static method는 독립적으로 사용할 수 있지만, class [namespace](./namespace.md)에 속하게 된다.
    - class나 instance에 접근할 수 있는 권한이 없으므로 수정하는 행위가 차단된다.
    - `@staticmethod` 데코레이터를 사용하여 정의한다.
+ 2) class method는 class 오브젝트에 관한 method로서 인스턴스에 대한 접근이 금지된다.
    - `@classmethod` 데코레이터를 사용하여  정의한다.
* 3) instance method는 인스턴스`self`에 접근할 수 있는 메서드이므로 instantiation이 반드시 필요하다.


```python
import numpy as np

class MyAccount:
    trans_cost = 10
    def __init__(self, name, account_number, balance=0):
        self.name = name
        self.account_number = account_number
        self.balance = balance
    # 1    
    def __repr__(self):
        return repr((self.name, self.account_number, self.balance))
    # 2
    @staticmethod    
    def interest_rate_year(r_month):
        return (1+r_month)**12 - 1
    # 3
    @classmethod
    def get_trans_cost(cls):
        return str(cls.trans_cost)
        
    
MyAccount.interest_rate_year(0.005)
```
    0.06167781186449828

```python
MyAccount.get_trans_cost()
```

    '10'


```python
myaccount = MyAccount('yang', '123-456-7890', 1000)
myaccount
```




    ('yang', '123-456-7890', 1000)



- `# 1` `__repr__()` : 인스턴스를 어떻게 나타낼지 정하는 인스턴스 메서드( `__str__()`도 있다.) 
- `# 2` 클래스 객체나 인스턴스 객체에 접근할 필요는 없지만, MyAccount에서 필요로 할 만한 기능이나 
    함께 묶어 객체화 하는 것이 자연스러운 메서드를 `@staticmethod`로 정의하면 좋을 것이다.
- `# 3` 인스턴스와는 별개지만, 클래스에 속하는 attributes에 관계된 메서드를 `@classmethod`로 
    정의하면 좋을 것이다.
    

    
