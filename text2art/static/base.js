function alertMessage(alertType, message, time = 1500) {
    let alertIdName = `#${alertType}-alert`;
    let myAlert = document.querySelector(alertIdName);
    let alertMessage = document.querySelector(alertIdName + '-message');

    myAlert.style.display = ""
    alertMessage.innerHTML = message
    myAlert.classList.toggle('myAlert-show')

    setTimeout(function () {
        // $(alertIdName).fadeOut()
        myAlert.classList.toggle('myAlert-show')
        myAlert.style.display = 'none'
    }, time)
}

function signupMessage(message) {
    let prompt = document.getElementById('signupPrompt');
    prompt.innerHTML = message;
    prompt.classList = "";
    prompt.style.color = 'red';
}

function signinMessage(message) {
    let prompt = document.getElementById('signinPrompt');
    prompt.innerHTML = message;
    prompt.classList = "";
    prompt.style.color = 'red';
}

function model2signup() {
    $('#signinModel').modal('hide');
    $('#signupModel').modal('show');
}

function model2signin() {
    $('#signupModel').modal('hide');
    $('#signinModel').modal('show');
}

function signin() {
    let accountInput = document.getElementById('signinAccount')
    let passwordInput = document.getElementById('signinPassword')
    let account = accountInput.value;
    let password = passwordInput.value;

    const passwordMinLength = 3

    accountInput.style.borderColor = '';
    passwordInput.style.borderColor = '';
    // 使用正则表达式对用户输入进行校验
    if (account.length === 0) {
        signinMessage('请填写账户!');
        accountInput.style.borderColor = 'red';
        return;
    } else if (password.length === 0) {
        signinMessage('请填写密码!')
        passwordInput.style.borderColor = 'red';
        return;
    } else if (!/^[a-zA-Z0-9]+$/.test(account)) {
        signinMessage('账户名中只能出现字母或数字!')
        accountInput.style.borderColor = 'red';
        return;
    } else if (password.length < passwordMinLength) {
        signinMessage('密码长度不得小于' + passwordMinLength + '位!')
        passwordInput.style.borderColor = 'red';
        return;
    } else if (!/^[0-9]+$/.test(password)) {
        signinMessage('密码中只能出现数字')
        passwordInput.style.borderColor = 'red';
        return;
    }

    const data = {
        "account": account,
        "password": password,
    }
    fetch('/index/signin', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
    })
        .then(response => {
            if (response.status === 200) {
                return response.json();
            } else {
                alert(response.status);
            }
        })
        .then(data => {
            //根据后端相应显示相应信息
            if (data.status === 0) {
                $('#signinModel').modal('hide');
                window.location.href = '/painter'
            } else if (data.status === 1) {
                signinMessage('用户不存在!');
                accountInput.style.borderColor = 'red';
            } else if (data.status === 2) {
                signinMessage('账户或密码错误！')
                passwordInput.style.borderColor = 'red';
            } else {
                alertMessage('danger', '未知状态 ' + data.status)
            }
        })
        .catch(error => console.error(error));
}

function signup() {
    let accountInput = document.getElementById('signupAccount')
    let passwordInput = document.getElementById('signupPassword')
    let account = accountInput.value;
    let password = passwordInput.value;

    const passwordMinLength = 3

    accountInput.style.borderColor = '';
    passwordInput.style.borderColor = '';
    // 使用正则表达式对用户输入进行校验
    if (account.length === 0) {
        signupMessage('请输入账户!');
        accountInput.style.borderColor = 'red';
        return;
    } else if (password.length === 0) {
        signupMessage('请输入密码!')
        passwordInput.style.borderColor = 'red';
        return;
    } else if (!/^[a-zA-Z0-9]+$/.test(account)) {
        signupMessage('账户名中只能出现字母或数字!')
        accountInput.style.borderColor = 'red';
        return;
    } else if (password.length < passwordMinLength) {
        signupMessage('密码长度不得小于' + passwordMinLength + '位!')
        passwordInput.style.borderColor = 'red';
        return;
    } else if (!/^[0-9]+$/.test(password)) {
        signupMessage('密码中只能出现数字')
        passwordInput.style.borderColor = 'red';
        return;
    }

    const data = {
        "account": account,
        "password": password,
    }
    fetch('/index/signup', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
    })
        .then(response => {
            if (response.status === 200) {
                return response.json();
            } else {
                alert(response.status);
            }
        })
        .then(data => {
            //根据后端相应显示相应信息
            if (data.status === 0) {
                $('#signupModel').modal('hide');
                window.location.href = '/painter'
                alertMessage('primary', '注册成功！')
            } else if (data.status === 1) {
                signupMessage('用户已存在!')
                accountInput.style.borderColor = 'red';
            } else {
                alertMessage('danger', '未知状态 ' + data.status)
            }
        })
        .catch(error => console.error(error));
}

function login() {
    let navSigninbtn = document.getElementById('navSigninBtn');
    let navSignupbtn = document.getElementById('navSignupBtn');
    let navLoginbtn = document.getElementById('navLoginBtn');

    navSigninbtn.style.display = 'none';
    navSignupbtn.style.display = 'none';
    navLoginbtn.style.display = '';
}


function logout() {
    let navSigninbtn = document.getElementById('navSigninBtn');
    let navSignupbtn = document.getElementById('navSignupBtn');
    let navLoginbtn = document.getElementById('navLoginBtn');

    navSigninbtn.style.display = '';
    navSignupbtn.style.display = '';
    navLoginbtn.style.display = 'none';

    // 清除所有Cookie
    let cookies = document.cookie.split(";");
    console.log(cookies)
    for (let i = 0; i < cookies.length; i++) {
        let cookie = cookies[i].split("=")[0];
        document.cookie = cookie + "=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    }

}

function passwordToggle(type = "signin") {
    // 切换密码的显示与隐藏
    let passwordSigninInput = document.getElementById(type + "Password");
    let eyeslash = document.getElementById(type + "PasswordEyeSlash");  // 密码隐层图标
    let eye = document.getElementById(type + "PasswordEye");    // 密码显示图标

    if (passwordSigninInput.type === "password") {
        passwordSigninInput.type = "text";
        eyeslash.style.display = "none";
        eye.style.display = "";
    } else {
        passwordSigninInput.type = "password";
        eye.style.display = "none";
        eyeslash.style.display = "";
    }
}
