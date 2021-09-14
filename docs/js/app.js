
$(document).ready(function() {
    var editor = CodeMirror.fromTextArea(document.getElementById("bibtex-law"), {
        lineNumbers: false,
        lineWrapping: true,
        readOnly:true
    });
    var editor_vlnce = CodeMirror.fromTextArea(document.getElementById("bibtex-vlnce"), {
        lineNumbers: false,
        lineWrapping: true,
        readOnly:true
    });
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
    

});