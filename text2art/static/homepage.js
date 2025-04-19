let gallery = document.querySelector('.gallery');
let galleryItems = document.querySelectorAll('.gallery-item');
let imageInfoItems = document.querySelectorAll('.image-info-item');
let numOfItems = gallery.children.length;
let itemWidth = 23; // 这边之所以为23是因为在CSS中有相关设置，CSS将每个图像的宽度设置为23
let selectedItem;

let featured = document.querySelector('img.featured-item');

let swiperLeftBtn = document.querySelector('.swiper-button-prev.swiper-button');
let swiperRightBtn = document.querySelector('.swiper-button-next.swiper-button');
let leftBtn = document.querySelector('.move-btn.left');
let rightBtn = document.querySelector('.move-btn.right');
let leftInterval;
let rightInterval;

let scrollRate = 0.4;   // 滚动速率
let left;

function selectItem(e) {
    if (e.target.classList.contains('active')) return;

    featured.src = e.target.src;

    for (let i = 0; i < galleryItems.length; i++) {
        if (galleryItems[i].classList.contains('active')) {
            galleryItems[i].classList.remove('active');
            imageInfoItems[i].style.display = 'none';
        }
    }

    let img = e.target;
    img.classList.add('active');
    let span = img.nextElementSibling;
    selectedItem = parseInt(span.innerHTML)
    imageInfoItems[selectedItem].style.display = '';
}

function galleryWrapLeft() {
    let first = gallery.children[0];
    gallery.removeChild(first);
    gallery.style.left = -itemWidth + '%';
    gallery.appendChild(first);
    gallery.style.left = '0%';
}

function galleryWrapRight() {
    let last = gallery.children[gallery.children.length - 1];
    gallery.removeChild(last);
    gallery.insertBefore(last, gallery.children[0]);
    gallery.style.left = '-23%';
}

function moveLeft() {
    left = left || 0;

    leftInterval = setInterval(function () {
        gallery.style.left = left + '%';

        if (left > -itemWidth) {
            left -= scrollRate;
        } else {
            left = 0;
            galleryWrapLeft();
        }
    }, 1);
}

function moveRight() {
    //Make sure there is element to the leftd
    if (left > -itemWidth && left < 0) {
        console.log('yeah')
        left = left - itemWidth;

        let last = gallery.children[gallery.children.length - 1];
        gallery.removeChild(last);
        gallery.style.left = left + '%';
        gallery.insertBefore(last, gallery.children[0]);
    }

    left = left || 0;
    console.log('fuck')

    leftInterval = setInterval(function () {
        gallery.style.left = left + '%';

        if (left < 0) {
            left += scrollRate;
        } else {
            left = -itemWidth;
            galleryWrapRight();
        }
    }, 1);
}

function stopMovement() {
    clearInterval(leftInterval);
    clearInterval(rightInterval);
}

// leftBtn.addEventListener('mouseenter', moveLeft);
leftBtn.addEventListener('mouseenter', moveRight);
leftBtn.addEventListener('mouseleave', stopMovement);
// rightBtn.addEventListener('mouseenter', moveRight);
rightBtn.addEventListener('mouseenter', moveLeft);
rightBtn.addEventListener('mouseleave', stopMovement);

swiperLeftBtn.addEventListener('mousedown', moveRight);
swiperLeftBtn.addEventListener('mouseup', stopMovement);
swiperLeftBtn.addEventListener('mouseleave', stopMovement);

swiperRightBtn.addEventListener('mousedown', moveLeft);
swiperRightBtn.addEventListener('mouseup', stopMovement);
swiperRightBtn.addEventListener('mouseleave', stopMovement);


//Start this baby up
(function init() {

    //Set Images for Gallery and Add Event Listeners
    for (let i = 0; i < galleryItems.length; i++) {
        galleryItems[i].addEventListener('click', selectItem);
    }

    selectedItem = 0;
    galleryItems[selectedItem].classList.add('active');
    imageInfoItems[selectedItem].style.display = '';
})();