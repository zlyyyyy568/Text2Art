var curSelectImage;
let candelete = false;

var get = {
    byId: function (id) {
        return typeof id === "string" ? document.getElementById(id) : id
    },
    byClass: function (sClass, oParent) {
        var aClass = [];
        var reClass = new RegExp("(^| )" + sClass + "( |$)");
        var aElem = this.byTagName("*", oParent);
        for (var i = 0; i < aElem.length; i++) reClass.test(aElem[i].className) && aClass.push(aElem[i]);
        return aClass
    },
    byTagName: function (elem, obj) {
        return (obj || document).getElementsByTagName(elem)
    }
};
/*-------------------------- +
  事件绑定, 删除
 +-------------------------- */
var EventUtil = {
    addHandler: function (oElement, sEvent, fnHandler) {
        oElement.addEventListener ? oElement.addEventListener(sEvent, fnHandler, false) : (oElement["_" + sEvent + fnHandler] = fnHandler, oElement[sEvent + fnHandler] = function () {
            oElement["_" + sEvent + fnHandler]()
        }, oElement.attachEvent("on" + sEvent, oElement[sEvent + fnHandler]))
    },
    removeHandler: function (oElement, sEvent, fnHandler) {
        oElement.removeEventListener ? oElement.removeEventListener(sEvent, fnHandler, false) : oElement.detachEvent("on" + sEvent, oElement[sEvent + fnHandler])
    },
    addLoadHandler: function (fnHandler) {
        this.addHandler(window, "load", fnHandler)
    }
};

/*-------------------------- +
  设置css样式
  读取css样式
 +-------------------------- */
function css(obj, attr, value) {
    switch (arguments.length) {
        case 2:
            if (typeof arguments[1] == "object") {
                for (var i in attr) i == "opacity" ? (obj.style["filter"] = "alpha(opacity=" + attr[i] + ")", obj.style[i] = attr[i] / 100) : obj.style[i] = attr[i];
            } else {
                return obj.currentStyle ? obj.currentStyle[attr] : getComputedStyle(obj, null)[attr]
            }
            break;
        case 3:
            attr == "opacity" ? (obj.style["filter"] = "alpha(opacity=" + value + ")", obj.style[attr] = value / 100) : obj.style[attr] = value;
            break;
    }
};

var globalDelLi;
var globalLiHover;

EventUtil.addLoadHandler(function () {
    var oMsgBox = get.byId("msgBox");
    var oUserName = get.byId("userName");
    var oConBox = get.byId("conBox");
    var oSendBtn = get.byId("sendBtn");
    var oMaxNum = get.byClass("maxNum")[0];
    var oCountTxt = get.byClass("countTxt")[0];
    var oList = get.byClass("list")[0];
    var oUl = get.byTagName("ul", oList)[0];
    var aLi = get.byTagName("li", oList);
    var aFtxt = get.byClass("f-text", oMsgBox);
    var aImg = get.byId('userPic');
    var bSend = false;
    var timer = null;
    var oTmp = "";
    var i = 0;
    var maxNum = 140;

    //禁止表单提交
    EventUtil.addHandler(get.byTagName("form", oMsgBox)[0], "submit", function () {
        return false
    });

    //为广播按钮绑定发送事件
    EventUtil.addHandler(oSendBtn, "click", fnSend);

    //为Ctrl+Enter快捷键绑定发送事件
    EventUtil.addHandler(document, "keyup", function (event) {
        event = event || window.event;
        event.ctrlKey && event.keyCode == 13 && fnSend()
    });

    //发送广播函数
    function fnSend() {
        var reg = /^\s*$/g;
        if (reg.test(oConBox.value)) {
            alert("\u968f\u4fbf\u8bf4\u70b9\u4ec0\u4e48\u5427\uff01");
            oConBox.focus()
        } else if (!bSend) {
            alert("\u4f60\u8f93\u5165\u7684\u5185\u5bb9\u5df2\u8d85\u51fa\u9650\u5236\uff0c\u8bf7\u68c0\u67e5\uff01");
            oConBox.focus()
        } else {
            var oLi = document.createElement("li");
            var oDate = new Date();
            let message = oConBox.value.replace(/<[^>]*>|&nbsp;/ig, "");

            let now = new Date();
            let year = now.getFullYear();
            let month = now.getMonth() + 1 < 10 ? `0${now.getMonth() + 1}` : now.getMonth() + 1;
            let date = now.getDate() < 10 ? `0${now.getDate()}` : now.getDate();
            let hour = now.getHours() < 10 ? `0${now.getHours()}` : now.getHours();
            let minute = now.getMinutes() < 10 ? `0${now.getMinutes()}` : now.getMinutes();
            let time = `${year}年 ${month}月${date}日 ${hour}:${minute}`;

            oLi.innerHTML = `
                                <img class="userPic" src="${aImg.src}" alt=""/>
                                <div class="content">
                                    <div class="userName">
                                        ${oUserName.innerHTML}
                                    </div>
                                    <div class="msgInfo">${message}</div>
                                    <div class="times">
                                        <span>${time}</span>
                                        <a class="del no-underline " style="display: none"
                                           href="javascript:void(0);">删除</a>
                                    </div>
                                </div>
            `
            //插入元素
            aLi.length ? oUl.insertBefore(oLi, aLi[0]) : oUl.appendChild(oLi);

            let commentNum = document.getElementById('commentNum')
            commentNum.innerHTML = parseInt(commentNum.innerHTML) + 1;

            //重置表单
            get.byTagName("form", oMsgBox)[0].reset();

            //将元素高度保存
            // var iHeight = oLi.clientHeight - parseFloat(css(oLi, "paddingTop")) - parseFloat(css(oLi, "paddingBottom"));
            var iHeight = oLi.clientHeight;
            var alpah = count = 0;
            css(oLi, {"opacity": "0", "height": "0"});
            timer = setInterval(function () {
                css(oLi, {"opacity": "0", "height": (count += 8) + "px"});
                if (count > iHeight) {
                    clearInterval(timer);
                    css(oLi, "height", iHeight + "px");
                    timer = setInterval(function () {
                        css(oLi, "opacity", (alpah += 10));
                        alpah > 100 && (clearInterval(timer), css(oLi, "opacity", 100))
                    }, 30)
                }
            }, 30);
            //调用鼠标划过/离开样式
            // liHover();
            //调用删除函数
            // delLi()

            data = {
                'message': message,
                'image_id': curSelectImage,
            }

            fetch('/gallery/sendComment', {
                method: 'POST',
                body: JSON.stringify(data),
            }).then(response => response.json())
                .then(data => {
                    console.log(data.message)
                })
        }
    };

    //事件绑定, 判断字符输入
    EventUtil.addHandler(oConBox, "keyup", confine);
    EventUtil.addHandler(oConBox, "focus", confine);
    EventUtil.addHandler(oConBox, "change", confine);

    //输入字符限制
    function confine() {
        var iLen = 0;
        for (i = 0; i < oConBox.value.length; i++) iLen += /[^\x00-\xff]/g.test(oConBox.value.charAt(i)) ? 1 : 0.5;
        oMaxNum.innerHTML = Math.abs(maxNum - Math.floor(iLen));
        maxNum - Math.floor(iLen) >= 0 ? (css(oMaxNum, "color", ""), oCountTxt.innerHTML = "\u8fd8\u80fd\u8f93\u5165", bSend = true) : (css(oMaxNum, "color", "#f60"), oCountTxt.innerHTML = "\u5df2\u8d85\u51fa", bSend = false)
    }

    //加载即调用
    confine();


    //li鼠标划过/离开处理函数
    function liHover() {
        for (i = 0; i < aLi.length; i++) {
            //li鼠标划过样式
            EventUtil.addHandler(aLi[i], "mouseover", function (event) {
                this.className = "hover";
                oTmp = get.byClass("times", this)[0];
                var aA = get.byTagName("a", oTmp);
                if (!aA.length) {
                    var oA = document.createElement("a");
                    oA.innerHTML = "删除";
                    oA.className = "del";
                    oA.href = "javascript:;";
                    oTmp.appendChild(oA)
                } else {
                    aA[0].style.display = "block";
                }
            });

            //li鼠标离开样式
            EventUtil.addHandler(aLi[i], "mouseout", function () {
                this.className = "";
                var oA = get.byTagName("a", get.byClass("times", this)[0])[0];
                oA.style.display = "none"
            })
        }
    }

    globalLiHover = liHover

    //删除功能
    function delLi() {
        let aA = get.byClass("del", oUl);

        for (i = 0; i < aA.length; i++) {
            aA[i].onclick = function () {
                var oParent = this.parentNode.parentNode.parentNode;
                var alpha = 100;
                var iHeight = oParent.offsetHeight;
                timer = setInterval(function () {
                    css(oParent, "opacity", (alpha -= 10));
                    if (alpha < 0) {
                        clearInterval(timer);
                        timer = setInterval(function () {
                            iHeight -= 10;
                            iHeight < 0 && (iHeight = 0);
                            css(oParent, "height", iHeight + "px");
                            iHeight == 0 && (clearInterval(timer), oUl.removeChild(oParent))
                        }, 30)
                    }
                }, 30);
                this.onclick = null
                let commentID = parseInt(this.getAttribute('commentID'))

                fetch('/gallery/deleteComment?commentID=' + commentID, {
                    method: 'GET',
                })
                    .then(response => {
                        let commentNum = document.getElementById('commentNum')
                        commentNum.innerHTML = parseInt(commentNum.innerHTML) - 1;
                    })
            }
        }
    }

    globalDelLi = delLi;

    //输入框获取焦点时样式
    for (i = 0; i < aFtxt.length; i++) {
        EventUtil.addHandler(aFtxt[i], "focus", function () {
            this.className = "active"
        });
        EventUtil.addHandler(aFtxt[i], "blur", function () {
            this.className = ""
        })
    }

    //格式化时间, 如果为一位数时补0
    function format(str) {
        return str.toString().replace(/^(\d)$/, "0$1")
    }


});


function createCommentLi(comment) {
    let str = `
        <li >
            <img class="userPic" src=${comment.imgsrc} alt=""/>
            <div class="content">
                <div class="userName">
                    ${comment.userName}
                </div>
                <div class="msgInfo">${comment.text}</div>
                <div class="times">
                    <span>${comment.time}</span>
                    <a class="del no-underline " style="display: none" commentID="${comment.id}"
                       href="javascript:void(0);">删除</a>
                </div>
            </div>
        </li>
    `

    return str
}

function getComments(image_id) {
    $('#commentModal').modal('show');
    curSelectImage = image_id;

    fetch('/gallery/getComments?imageID=' + image_id, {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            let ul = document.getElementById('commentUL')
            ul.innerHTML = ""
            for (let i = 0; i < data.comments.length; i++) {
                ul.innerHTML += createCommentLi(data.comments[i])
            }
            let commentNum = document.getElementById('commentNum')
            commentNum.innerHTML = data.comments.length;

            if (candelete) {
                globalLiHover()
                globalDelLi()
            }

        });
}