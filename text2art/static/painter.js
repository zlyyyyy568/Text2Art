let timerUpdateImage = null;

function updateImage() {
    // 向服务器发送请求获取模型运行状态，并一次更新界面
    fetch('/painter/generate-image', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            // 获取线程控制块中的各种标识
            let isStart = data.isStart;
            let isInitial = data.isInitial;
            let isImgGenerateStart = data.isImgGenerateStart;
            let isVideoGenerateStart = data.isVideoGenerateStart;
            let isInterrupt = data.isInterrupt;
            let isFinish = data.isFinish;

            if (isStart) {
                // 后端接收到前端运行模型请求，开始运行模型

                // 若后端返回图片，由于前端返回的图片是base64格式的，故需要将前端界面的图片src属性设置为一个以'data:image/png;base64,'开头，后跟图片数据的字符串
                // 这样设置图片的src属性可以显示一个base64编码的PNG图片。
                // 若后端没有返回图片，则前端根据用户选择的长宽比设置背景图像
                if (data.image === "") {
                    setImageSrc(aspectRatioSelect.value);
                } else {
                    document.getElementById('my-image').src = 'data:image/png;base64,' + data.image;
                }

                // 根据AIPainter的当前迭代状态更新进度条
                let progressBar = document.querySelector('.progress-bar');
                let progressDiv = document.querySelector('.progress');
                let progressLabel = document.getElementById('progressLabel');
                let progressSpinner = document.getElementById('progressSpinner');
                let newProgress = data.curIter;
                let maxProgress = progressDiv.ariaValueMax;

                progressBar.style.width = newProgress / maxProgress * 100 + '%';
                progressBar.setAttribute('aria-valuenow', newProgress);
                progressSpinner.style.display = "none";
                progressLabel.innerHTML = '[ ' + newProgress + ' / ' + maxProgress + ' ]';

                if (isInitial) {
                    // AIPainter正在初始化中
                    progressSpinner.style.display = "inline-block";
                    progressLabel.innerHTML = "正在初始化模型...";
                } else {
                    if (isFinish) {
                        // AIPainter运行结束
                        clearInterval(timerUpdateImage);    // 清除定时器
                        progressSpinner.style.display = "none";
                        progressLabel.innerHTML = "成功生成视频！";
                        if (isInterrupt) {
                            // AIPainter由于中断导致运行结束
                            reset2Initial("图像生成终止!");   // 将界面恢复为默认状态
                        }
                    } else if (isVideoGenerateStart) {
                        // AIPainter正在生成视频
                        progressSpinner.style.display = "inline-block";
                        progressLabel.innerHTML = "正在生成视频...";
                    } else {
                        // AIPainter正在迭代生成图像

                    }
                }
            } else if (isFinish) {
                clearInterval(timerUpdateImage);
                progressSpinner.style.display = "none";
                progressLabel.innerHTML = "成功生成视频！";
                if (isInterrupt) {
                    reset2Initial("图像生成终止!");
                }
            }

        });
}

function startGeneration() {
    let progressDiv = document.querySelector('.progress');
    let maxiter = document.querySelector('#max-iterations').value;
    progressDiv.setAttribute('aria-valuemax', maxiter);

    let progressLabel = document.getElementById('progressLabel');
    let progressSpinner = document.getElementById('progressSpinner');
    progressSpinner.style.display = "inline-block";
    progressLabel.innerHTML = "正在初始化模型...";

    // 设置一个定时器，定期向后端发出更新请求，获取AIPainter的最新状态
    timerUpdateImage = setInterval(updateImage, 2000);
}

function continnueGeneration() {
    let progressDiv = document.querySelector('.progress');
    let maxiter = document.querySelector('#max-iterations').value;
    progressDiv.setAttribute('aria-valuemax', maxiter);

    let progressLabel = document.getElementById('progressLabel');
    let progressSpinner = document.getElementById('progressSpinner');
    // progressSpinner.style.display = "inline-block";
    progressSpinner.style.display = "";
    progressLabel.innerHTML = "正在连接服务器,从先前进度中恢复中...";

    // 设置一个定时器，定期向后端发出更新请求，获取AIPainter的最新状态
    timerUpdateImage = setInterval(updateImage, 2000);
}

function interruptGeneration() {
    fetch('/painter/interrupt-generate', {
        method: 'GET',
    })
        .then(response => {
            console.log('终止生成')
            alertMessage('danger', '图像生成终止！')
        });
}

function reset2Initial(labelMessage = "没有正在运行的任务") {
    // document.getElementById('my-image').src = "/static/resource/default-image-square.png";
    setImageSrc(aspectRatioSelect.value);

    let progressBar = document.querySelector('.progress-bar');
    let progressSpinner = document.getElementById('progressSpinner');


    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', 0);
    progressSpinner.style.display = "none";
    // 图像生成终止!
    document.getElementById('progressLabel').innerHTML = labelMessage;
}

function watchVideo(path) {
    let prompt = document.getElementById('prompt').value;

    document.getElementById("videoModalTitle").textContent = prompt;
    // document.querySelector('#myVideo source').setAttribute("src", "static/results/videos/temp.mp4");
    // document.querySelector('#myVideo source').setAttribute("src", path);
    document.querySelector('#myVideo source').setAttribute("src", path);

    let modalVideo = document.querySelector('#myVideo');
    modalVideo.load();
}

function downloadVideo() {
    let videoPath = document.querySelector('#myVideo source').src;
    let parma = "filePath=" + videoPath;

    window.location.href = "/download?" + parma;
}

function saveImage() {
    fetch('/painter/saveImage', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            if (data.success === true) {
                alertMessage('primary', '成功保存图片！')
                let progressLabel = document.getElementById('progressLabel');
                let progressSpinner = document.getElementById('progressSpinner');
                progressSpinner.style.display = "none";
                progressLabel.innerHTML = "成功确认图片!";
            } else if (data.error === 'notFound') {
                alertMessage('warning', '你没有待确认的图片！')
            }

        })
        .catch(error => {
            alertMessage('danger', '糟糕，有什么东西出错了...')
            console.error(error)
        });
}

function clearImage() {
    fetch('/painter/clearImage', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => {
            if (data.success === true) {
                alertMessage('primary', '成功清除图片！')
                setImageSrc(aspectRatioSelect.value)
                let progressLabel = document.getElementById('progressLabel');
                let progressSpinner = document.getElementById('progressSpinner');
                progressSpinner.style.display = "none";
                progressLabel.innerHTML = "成功确认图片!";
            } else if (data.error === 'notFound') {
                alertMessage('warning', '你没有待确认的图片！')
            }

        })
        .catch(error => {
            alertMessage('danger', '糟糕，有什么东西出错了...')
            console.error(error)
        });
}


const widthInput = document.getElementById('width-input');
const heightInput = document.getElementById('height-input');
const imageResolution = document.getElementById('image-resolution');
const aspectRatioSelect = document.getElementById('AspectRatioSelect');
const f = 16

widthInput.addEventListener('input', () => {
    checkImageResolution(widthInput.value)
    heightInput.value = width2height(widthInput.value, heightInput.value)

    const width = widthInput.value;
    const height = heightInput.value;

    imageResolution.textContent = width + ' x ' + height;
});

heightInput.addEventListener('input', () => {
    checkImageResolution(heightInput.value)
    widthInput.value = height2width(widthInput.value, heightInput.value)

    const width = widthInput.value;
    const height = heightInput.value;

    imageResolution.textContent = width + ' x ' + height;
});


aspectRatioSelect.addEventListener('change', () => {
    // 添加事件，当用户在前端页面点击不同图像长宽比时，自动更新前端界面以适合图像

    // 不知道为什么，明明之前已经定义了aspectRatioSelect，如果直接用的话会报错，搞不懂。。。
    // let aspectRatioSelect = document.getElementById("AspectRatioSelect");
    // let selectedValue = aspectRatioSelect.value;

    // 基于width调整height
    const width = widthInput.value;
    heightInput.value = width2height(widthInput.value, heightInput.value)


    let selectedValue = aspectRatioSelect.value;

    if (selectedValue === 'customize') {
        setImageSrc('square');
    } else if (selectedValue === 'widescreen') {
        // 16:9
        setImageSrc('widescreen');
    } else if (selectedValue === 'portrait') {
        // 3:4
        setImageSrc('portrait');
    } else if (selectedValue === 'square') {
        // 1:1
        setImageSrc('square');
    } else {
        // handle custom aspect ratio
    }

});


function checkImageResolution(val) {
    let val_integer = parseInt(val / f) * f;
    if (val !== val_integer) {
        alertMessage('warning', '输入必须是 ' + f + ' 的整数倍！')
    }
}

function height2width(width, height) {
    // 根据高度获取对应长宽比的宽度
    // let aspectRatioSelect = document.getElementById("AspectRatioSelect");
    let selectedValue = aspectRatioSelect.value;


    if (selectedValue === 'customize') {
        return width;
    } else if (selectedValue === 'widescreen') {
        // 16:9
        return parseInt(height / 9 * 16);
    } else if (selectedValue === 'portrait') {
        // 3:4
        return parseInt(height / 4 * 3);
    } else if (selectedValue === 'square') {
        // 1:1
        return height
    } else {
        // handle custom aspect ratio
    }
}

function width2height(width, height) {
    // 根据宽度获取对应长宽比的高度
    // let aspectRatioSelect = document.getElementById("AspectRatioSelect");
    let selectedValue = aspectRatioSelect.value;

    if (selectedValue === 'customize') {
        return height;
    } else if (selectedValue === 'widescreen') {
        // 16:9
        return parseInt(width / 16 * 9);
    } else if (selectedValue === 'portrait') {
        // 3:4
        return parseInt(width / 3 * 4);
    } else if (selectedValue === 'square') {
        // 1:1
        return width
    } else {
        // handle custom aspect ratio
    }
}

var src = ''

function setImageFrame(type) {
    // 设置图像边框已适应不同长宽比
    if (type === 'customize')
        type = 'square';

    const element = document.querySelector('.border-before');
    if (type === 'square') {
        element.style.paddingBottom = '100%';
    } else if (type === 'widescreen') {
        let ratio = 12 / 16 * 100
        element.style.paddingBottom = ratio + '%';
    } else if (type === 'portrait') {
        let ratio = 4 / 3 * 100
        element.style.paddingBottom = ratio + '%';
    } else {
        alert("错误的图片地址: " + src)
    }
}

function setImageSrc(type) {
    // 根据不用长宽比设置相应占位背景
    if (type === 'customize')
        type = 'square';

    src = `/static/resource/default-image-${type}.png`
    console.log(src)
    setImageFrame(type)


    document.getElementById('my-image').src = src;
}
