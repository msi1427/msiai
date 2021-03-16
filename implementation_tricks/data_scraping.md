# Download From Google Images

Insert the following code in the Javascript console after searching for whatever you want in Google Images to get the download link for images in a file. 

```javascript
# For Windows
urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```