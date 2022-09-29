# 1. Sphinx for rendering Markdown to HTML

## 1.1 Workflows and Goal
markdown 쉽고 `LaTex` 수준으로 정갈한 스타일의 mark up language 이다. 빠르면서 깔끔하게 정리할 수 있어 많이 이용하게 되었다. 그래서 markdown으로 작성된 문서를 번잡한 과정을 거치지 않고 배포까지 할 수 있게 되면 금상첨화일 것이란 생각이 자연스레 들었다. sphinx와 github pages가 그 주인공이다.  

- markdown 문서는 `.md` text문서와 함께 `.png`와 같은 문서에 쓰이는 이미지도 함께 보관하는 것이 편리하다. 그래서 항상 `.md` 파일들과 `images/` 폴더는 따라다닌다.  
- 문서들의 hierachy는 `index.rst` 등의 `reStructuredText` 마컵 언어로 작성된다. (약간의 학습이 필요하다.)  
- python `.py` 파일을 autodoc하는 과정은 현재로서는 필요하지 않아서 생략한다.  

- base `.md`문서들이 보관되는 폴더를 만들고 그 안에서 sphinx를 시작한다.  
- 그리고 나서, sphinx 설정 (path, 테마, 확장 등)몇가지를 한다.  
- `.rst`파일에서 hierachy를 정해주고 나서  
- sphinx로 문서들을 `.html`파일로 빌드한다.  
- 다음은 ghihub 저장소에 올린다음   
- github pages / actions 로 배포하면 끝이다.  

## 1.2 손이 좀 간다.

- 번거로움이 줄어든 것은 맞지만, 그래도 파일릉 옮기고 변환하고 하는 수고는 여전하다.   
- 문서가 추가되거나 변경될 때마다 위 과정을 반복해야 한다.   
- 체계적으로 정리되고 hypter texting이 되는 정적 포스팅이 된다는 점은 매력적이다.    

## 1.3 Local에서 하게 되는 구체적 과정과 주요 명령어

### 1.3.1 sphinx설치 및 준비단계
- 우선 파이썬의 경우 가상환경을 준비한다.  
    - 프로젝트에 관한 문서로 체워질 경우 프로젝트에서 사용되는 가상환경을 포함하게 될 것이다.(autodoc을 사용하려는 경우)  
    - 그 가상환경에서 pip 으로 sphinx와 파서, 테마 등을 설치한다.  
    ```bash
    pip install sphinx
    pip install myst_parser
    pip install sphinx_rtd_theme
    ```
- 배포용 git branch를 따로 두는 (예를 들어 gh-pages)예제들이 많다.  

- 프로젝트 폴더 안에서 `docs/` 라는 sphinx가 기동할 최상위 폴더를 만든다. 그 안에서 sphinx tool을 다음 명령으로 설치한다.  
    ```bash
    sphinx-quickstart
    ```

- 다음의 선택사항을 확인하는 메시지가 출력되고 적당히 응답한다.

```bash
You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]: 

The project name will occur in several places in the built documentation.
> Project name:
> Author name(s): 
> Project release []:


> Project language [en]: 
```

### 1.3.2 `.md` 문서(폴더)를 이동/작성 하고 hierachy룰 기록한다.
- `docs/source/` 에 `.md` 문서와 관련 `images/`폴더를 넣는다.
- 문서들이 어떤 주제로 묶이고 상하관례가 어떻게 되는지를 `.rst`파일로 기록하여 저장한다.
    - 3 space indentation이므로 주의한다.

### 1.3.3 설정하고서 local computer에서 html파일들로 build한다
- `conf.py` 가 설정파일이다. 
    - 첫부분에서 path 관련 코멘트를 해제하고 `(.)` 를 `(../..)`으로 즉, `doc/`폴더를 최상위 폴더로 만든다.
    - 다음처럼 각 항목을 채운다.  markdown 파서와 theme에 관한 확장들이다.
    ```python
    extensions = [
    'sphinx_rtd_theme',
    'myst_parser',  
    ]
    source_suffix = { # https://www.sphinx-doc.org/en/master/usage/markdown.html
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
    }
    ```

    ```python
    html_theme = 'sphinx_rtd_theme'
    ```

### 1.3.4 github pages에서 작동할 수 있게 두 가지 파일 추가
- `source/` 폴더에 다음 두 개의 파일을 추가한다.   
    - `.nojekyll` jekyll을 사용하지 않곘다는 목적의 빈 파일이다.  
    - `index.html` 시작페이지를 rediret할 목적으로 다음 스크립트로 채운다.  
    ```html
    <html>  
    <head>  
        <meta http-equiv="refresh" content="0; url=./build/html/index.html" />  
    </head>  
    </html>  
    ```  

## 1.4 github 에 올리고 배포하기까지

- 배포용 브랜치 `gh-pages`로 올린다.  
- settings로 클릭해서 들어가면 사이드 바에서 pages항목으로 클릭해 이동한다.  
- Source 항목에서 `Deploy from a branch`, Branch 항목에서 `gh-pages` , `/docs/` 로 설정하고 `Save`한다.  
- Actions 탭을 클릭하면 배포과정을 볼 수 있고 완료할 때까지 기다려야 한댜.  
- 일단락된 [사이트](https://phycosmos.github.io/docs/build/html/index.html)


## 1.5 Some issues and fixing them.
- [bullet list 가 뷸렛 없이 랜더링 되는 경우](https://stackoverflow.com/questions/67542699/readthedocs-sphinx-not-rendering-bullet-list-from-rst-file),
    - docutils=0.16 으로 다운 그레이드하면 해결된다. 즉,
    ```bash
    conda install docutils=0.16
    ```