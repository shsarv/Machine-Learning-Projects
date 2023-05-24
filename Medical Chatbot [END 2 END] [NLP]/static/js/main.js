$(document).ready(function () {
  symptoms = JSON.parse(symptoms);
  let input = $("#message-text");
  let sendBtn = $("#send");
  let startOverBtn = $("#start-over");
  let dataList = $("#symptoms-list");
  let chat = $("#conversation");

  // Handler for any input on the message input field
  input.on("input", function () {
    let insertedValue = $(this).val();
    $("#symptoms-list").empty();

    if (insertedValue.length > 1) {
      ssymptoms = $.fn.getSuggestedSymptoms(insertedValue);
      if (ssymptoms.length === 0) {
        $(".symptoms-list-container ").slideUp();
      } else {
        for (let i = 0; i < ssymptoms.length; i++) {
          var li = document.createElement("li");
          li.textContent = ssymptoms[i];
          dataList.append(li);
        }
        $(".symptoms-list-container ").slideDown();
      }
    } else {
      $(".symptoms-list-container ").slideUp();
    }
  });

  startOverBtn.on("click", function () {
    $.fn.startOver();
  });

  sendBtn.on("click", function () {
    $.fn.handleUserMessage();
  });

  // Handler for click on one of the suggested symptoms
  dataList.on("click", "li", function () {
    input.val($(this).text());
    $(".symptoms-list-container ").slideUp();
  });
  //todo: blur on input - does not work with suggestion item clicks

  input.on("blur", function () {
    $(".symptoms-list-container ").slideUp();
  });

  input.on("keypress", function (e) {
    if (e.which == 13) {
      $.fn.handleUserMessage();
    }
  });

  // Handler function for sending a message
  $.fn.handleUserMessage = function () {
    if (input.val()) {
      $.fn.appendUserMessage();
      $.fn.getPredictedSymptom();
      input.val("");
      chat.animate({
        scrollTop: $("#conversation .message-body:last-child").position().top,
      });
    }
  };

  $.fn.startOver = function () {
    $.fn.getPredictedSymptom(true);
    $("#conversation").empty();
    const text =
      "Welcome! I'm Medical Chatbot, but you can call me Meddy. What symptoms are you currently experiencing? When you've entered all of your symptoms, please write '<b>Done</b>'. Make sure you enter as much symptoms as possible so the prediction can be as correct as possible.";
    $("#conversation").append(
      `<div class="row message-previous"><div class="col-sm-12 previous"></div></div><div class="row message-body"><div class="col-sm-12 message-main-receiver"><div class="receiver"><div class="message-text">${text}</div></div></div></div>`
    );
    input.val("");
  };

  // Creates the newly sent message element
  $.fn.appendUserMessage = function () {
    var text = input.val();
    $("#conversation").append(
      `<div class="row message-body"><div class="col-sm-12 message-main-sender"><div class="sender"><div class="message-text">${text}</div></div></div></div>`
    );
  };

  $.fn.appendBotMessage = function (text) {
    $("#conversation").append(
      `<div class="row message-body"><div class="col-sm-12 message-main-receiver"><div class="receiver"><div class="message-text">${text}</div></div></div></div>`
    );
  };

  // Retreives prediction to show as bot message
  $.fn.getPredictedSymptom = function (again) {
    var text = input.val();
    if (again) text = "done";

    $.ajax({
      url: "http://127.0.0.1:5000/symptom",
      data: JSON.stringify({ sentence: text }),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      type: "POST",
      success: function (response) {
        console.log(response);
        if (!again) $.fn.appendBotMessage(response);
      },
      error: function () {
        console.log("Error");
      },
    });
  };

  $.fn.getSuggestedSymptoms = function (val) {
    let suggestedSymptoms = [];
    $.each(symptoms, function (i, v) {
      if (v.includes(val)) {
        suggestedSymptoms.push(v);
      }
    });
    return suggestedSymptoms.slice(0, 3);
  };
});
