from sympy import (together, expand_complex, re, im, symbols, sympify,
    fraction, random_poly, sqf_part, gcd, lambdify, count_roots)

import numpy as np

from ..partialfrac import (t, allroots, thetas_alphas, thetas_alphas_to_expr,
    thetas_alphas_to_expr_complex, multiply_vector, customre)

n0 = symbols("n0", commutative=False)

def test_re_form():
    theta, alpha = symbols('theta, alpha')

    # Check that this doesn't change
    re_form = together(expand_complex(re(alpha/(t - theta))))
    assert re_form == (t*re(alpha) - re(alpha)*re(theta) -
        im(alpha)*im(theta))/((t - re(theta))**2 + im(theta)**2)

def test_allroots():
    # The rational function for degree 14 (prec=200)
    rat_func = sympify("(0.00000000000000000000000000041610013267680029528207499294424765814230151693834355411329381265006523459117230810479671980683866795673273632410330697354294409173232218813286143983868626253200155264994291921765203407267953152937058*t**14 - 0.00000000000000000000000377945323226709559694881101412595833466733624098302130643765177297549410491130489758476352772424958657745649985632149760654108316642962737165308057893935474692159122704791299213504744117706727097103*t**13 + 0.0000000000000000000056743670690330211958528356245306218648596518507488868609740226265011028708799769442380748354722826697215151311938483137291895625346545527891089924424411950997814572880910772358179618181743271947192744*t**12 - 0.0000000000000000033490171642170337489476483201547760775801561338961536903867154764265252672723753736290656486372345493075806122339657311965041895210482669724159623054416555702135014745759899700241199613981832015131967*t**11 + 0.000000000000001031012024963589780494347450494469914513309012322187086908113587030069872392401167002032361442371293275441976658212020477902165492558598666575960183091627341529823291449548178691325540927775878519095*t**10 - 0.00000000000019038591551025381665770379443609822366958724787778603816174389152153985569809764860302714423958935769613930886318328707282144521123416852145957992387999342713612032448824462163717985349856418887076421*t**9 + 0.000000000022849466635625509297782660989043204175886877257930759104470573118818209913052359166557843963201380130322290437020635175189129943415548578670954620086061792945424811777545207190661910100186982599078072*t**8 - 0.0000000018710223403465319987756758293306995709040697027221257329668078424591828113514747172479139688566890399472779688084549183920550396528279958427154763082513955918991672522958075213081879222826945261599508*t**7 + 0.00000010753187241762202746310450538934168613647817962524584961539915521930534383294853436829736498866457726054301856750299957698702854802932424275866290132293375082546585391682182318194520168551920423723243*t**6 - 0.0000043932761847806353276387222930098543450448974426503860062146526177605573818321831226504707461940359452161173095756770219014380246477951028077615962851743870321714471417119398278435423659553287344588111*t**5 + 0.00012734060684477834083893287425774209688011459840530930555717393060127049131989480579702745633229401763133892437617677800502408126113929136796486350264087356530539950809206807623611422934737503393301725*t**4 - 0.0025674425707902795730052723775680757187643386497164615125625547276799694382972544687443469801597632471308014042988393917904784802244249807212772123875585709329120572213784827300582274653220846471175742*t**3 + 0.034346972429252957161781653254459748124408267753288586328059453249056353034909403477212945622625320790267991512929524982607837276901282279468143112145914119795345690999025812897554582217358026787779485*t**2 - 0.27495599893993723128579991470979893780340145314063359381333206327371008015966563603449845678528043720768970737043737126558662975662923222244591395333902877524989375532021584103657849263855445663772808*t + 0.99999999999998167825621745958724844498245785243440872246997533333258161511410462628986744775121430638146283341836441479725132395162291463493988787827952756159667420482776106481433061650998672167570152)/(0.000000000000022710727625899900736639712742230262681012667462506304567144990897211911319763414687524273266701143110949302758330270167017813343148216312019170895784516633693561521998314350330069949570087366058032179*t**14 + 0.00000000000018921825619816040159907167022170105225585096872877673365980170501644130346080005909718893028091513777901533223902794281612808574140233988243236665867235329349587922609438809291197675387886822751627778*t**13 + 0.000000000012922822504467751142249299128115810954745070989745159313903014663967573892023183318656086391929119118292319109744980705409282107171658014762525420195015729956111735804263924100894800704597370375332887*t**12 + 0.00000000022472046467801857194300222740737195359062342716152199464902714290332834772321830419570280720976240865351181483300666560686634230379486145091844368844120360940092696175107818422827145258147276102361273*t**11 + 0.0000000048361843519832766388699493334045457225846727460322341138148449513139371607278723742186561699357605886385321786816589681632015387479142760930780407369310259224696173607801660762550730892297542033292215*t**10 + 0.000000080164052804759260531194459250275493894886010619764008351229291773071096261141811030620811410418728397391496461398766660652290575420937723645986684863398674973864626779366227127748036975191370976025427*t**9 + 0.0000011820298688854366697091577678949005806225282374279296460769042239639706165089625186706120394080653086569598914072457795895652205537291142192007017244106252451658922773770707596184858349460272598778996*t**8 + 0.000014908242237501135776503697901454105995074083378539243528806794109279484737376246523112843387554335018203486733697200509121603779983884241759028121119587704038295248900564437768718182261897340060096951*t**7 + 0.00016019218139205361453263922521232304971091995309393646119892238771814811156669222681928834878695605123127246992906338628046114029080259772428585261626366778937669585027331372119515267732298588220893931*t**6 + 0.001440553201917513102347480975935821470279275643748716489961983189427552682519213027972074267191526115687019253284032034535264582543735866826371462017406815737461262089695993267788839259541482779193108*t**5 + 0.01057405193647140271493174754948548438366062908830611709818356615838013184544233452878012084426887028406468781286541868730842173296561053238028901760563133099543012640533359019400611195549619733155274*t**4 + 0.060968196803506752124801588821965963591918947334946645054226140523076768775268308762513161265974970033209157273874477946771693248524307489116740979343580535039467361328062374494968473999138781014690633*t**3 + 0.25939097352682523362026784145948440878999306102781353930241675031360643928290818319695549634034466233021745574707991159818934961175123508232645772977199405943861729655381338474267315285698461370424366*t**2 + 0.72504400105794982800729693386049169070135183597227909285008688293747682637373140795446330817976221979059869956234370926079574649173651327145637768231993803890421024746968517841311285386973926251865771*t + 1.0)", locals=globals())
    # locals=globals() here will ensure it uses the t with the real assumption
    num, den = fraction(rat_func)
    roots = allroots(den, 14, 200)
    assert len(roots) == 14
    for i in roots:
        assert abs(den.subs(t, i).expand()) < 1e-194

    roots2 = allroots(den, 14, 200)
    assert roots == roots2

def test_exprs():
    num = random_poly(t, 10, -10, 10)
    while True:
        # Make sure den doesn't have repeated roots
        den = random_poly(t, 10, -10, 10)
        if sqf_part(den) != den:
            continue
        # Make sure den doesn't share any roots with num
        if gcd(num, den) != 1:
            continue
        # The complex formula assumes no real roots
        if count_roots(den) != 0:
            continue
        break

    rat_func = num/den
    thetas, alphas, alpha0 = thetas_alphas(rat_func, 15)
    part_frac = thetas_alphas_to_expr(thetas, alphas, alpha0)
    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)

    lrat_func = lambdify(t, rat_func, 'numpy')
    lpart_frac = lambdify(t, part_frac, 'numpy')
    lpart_frac_complex = lambdify(t, part_frac_complex, ['numpy', {'customre': np.real}])

    vals = np.random.random(10)

    np.testing.assert_allclose(lpart_frac(vals), lrat_func(vals))
    np.testing.assert_allclose(lpart_frac_complex(vals), lrat_func(vals))


# Used by test_multiply_vector. Different forms tested as correct by test_forms
rat_func4 = sympify("""(0.0000016765299308108737248115802898*t**4 -
    0.000449815029070811764481317961325*t**3 +
    0.0184005623076780392150344724797*t**2 -
    0.240254024325459538837623100106*t +
    0.999913477593047111476517762205)/(0.0193768295387776807297603967312*t**4
    + 0.045750548404322635676959079107*t**3 +
    0.291753976337465123448099119431*t**2 + 0.756683068883297082135320717398*t
    + 1.0)""", locals=globals())

part_frac4 = sympify("Add(Mul(Integer(2), Add(Mul(Integer(-1), Float('0.073395957163942207458442844011801', prec=30), Symbol('t', real=True)), Float('-1.6191559365989838437574840737357', prec=30)), Pow(Add(Pow(Add(Symbol('t', real=True), Float('-0.3678453861815398380437964998596', prec=30)), Integer(2)), Float('13.3818514358464994390440914505336', prec=30)), Integer(-1))), Mul(Integer(2), Add(Mul(Float('0.061686779567832924167854223579848', prec=30), Symbol('t', real=True)), Float('2.36598694484885076112187804632686', prec=30)), Pow(Add(Pow(Add(Symbol('t', real=True), Float('1.54839322329712217439052993875215', prec=30)), Integer(2)), Float('1.42044193610767940377598420545949', prec=30)), Integer(-1))), Float('0.0000865224069528885234822377947049701', prec=30))")

part_frac_complex4 = sympify("""Add(Mul(Integer(2),
customre(Add(Mul(Add(Float('0.061686779567832924167854223579848', prec=30),
Mul(Integer(-1), Float('1.90504097930308129535280142244488', prec=30), I)),
Pow(Add(Symbol('t', real=True), Float('1.54839322329712217439052993875215',
prec=30), Mul(Integer(-1), Float('1.19182294662742561401495340245672',
prec=30), I)), Integer(-1))),
Mul(Add(Float('-0.073395957163942207458442844011801', prec=30),
Mul(Float('0.449999922474062719432845030483482', prec=30), I)),
Pow(Add(Symbol('t', real=True), Float('-0.3678453861815398380437964998596',
prec=30), Mul(Integer(-1), Float('3.65812129867866730352792994532021',
prec=30), I)), Integer(-1)))))),
Float('0.0000865224069528885234822377947049701', prec=30))""", locals=globals())

def test_forms():
    """
    Test the expressions used by test_multiply_vector
    """
    thetas, alphas, alpha0 = thetas_alphas(rat_func4, 30)

    part_frac = thetas_alphas_to_expr(thetas, alphas, alpha0)
    assert part_frac == part_frac4

    part_frac_complex = thetas_alphas_to_expr_complex(thetas, alphas, alpha0)
    assert part_frac_complex == part_frac_complex4

    lrat_func = lambdify(t, rat_func4, 'numpy')
    lpart_frac = lambdify(t, part_frac, 'numpy')
    lpart_frac_complex = lambdify(t, part_frac_complex, ['numpy', {'customre': np.real}])

    vals = np.random.random(10)

    np.testing.assert_allclose(lpart_frac(vals), lrat_func(vals))
    np.testing.assert_allclose(lpart_frac_complex(vals), lrat_func(vals))

def test_multiply_vector():
    rat_func_vec = multiply_vector(rat_func4, n0)

    assert rat_func_vec == sympify("""(0.0000016765299308108737248115802898*t**4*n0 -
    0.000449815029070811764481317961325*t**3*n0 +
    0.0184005623076780392150344724797*t**2*n0 -
    0.240254024325459538837623100106*t*n0 +
    0.999913477593047111476517762205*n0)/(0.0193768295387776807297603967312*t**4
    + 0.045750548404322635676959079107*t**3 +
    0.291753976337465123448099119431*t**2 + 0.756683068883297082135320717398*t
    + 1.0)""", locals=globals())

    rat_func_horner_vec = multiply_vector(rat_func4, n0, horner=True)

    assert rat_func_horner_vec == sympify("""(t*(t*(t*(0.0000016765299308108737248115802898*t*n0 -
    0.000449815029070811764481317961325*n0) +
    0.0184005623076780392150344724797*n0) -
    0.240254024325459538837623100106*n0) +
    0.999913477593047111476517762205*n0)/(t*(t*(t*(0.0193768295387776807297603967312*t
    + 0.045750548404322635676959079107) + 0.291753976337465123448099119431) +
    0.756683068883297082135320717398) + 1.0)""", locals=globals())

    part_frac_vec = multiply_vector(part_frac4, n0)

    assert part_frac_vec == sympify("""Add(Mul(Float('0.0000865224069528885234822377947049701', prec=30), Symbol('n0', commutative=False)), Mul(Pow(Add(Pow(Add(Symbol('t', real=True), Float('1.54839322329712217439052993875215', prec=30)), Integer(2)), Float('1.42044193610767940377598420545949', prec=30)), Integer(-1)), Add(Mul(Float('0.123373559135665848335708447159696', prec=30), Symbol('t', real=True), Symbol('n0', commutative=False)), Mul(Float('4.73197388969770152224375609265371', prec=30), Symbol('n0', commutative=False)))), Mul(Pow(Add(Pow(Add(Symbol('t', real=True), Float('-0.3678453861815398380437964998596', prec=30)), Integer(2)), Float('13.3818514358464994390440914505336', prec=30)), Integer(-1)), Add(Mul(Integer(-1), Float('0.146791914327884414916885688023602', prec=30), Symbol('t', real=True), Symbol('n0', commutative=False)), Mul(Integer(-1), Float('3.2383118731979676875149681474714', prec=30), Symbol('n0', commutative=False)))))""", locals=globals())

    part_frac_complex_vec = multiply_vector(part_frac_complex4, n0)

    assert part_frac_complex_vec == sympify("""Add(Mul(Float('0.0000865224069528885234822377947049701', prec=30), Symbol('n0', commutative=False)), Mul(Integer(2), customre(Add(Mul(Pow(Add(Symbol('t', real=True), Float('1.54839322329712217439052993875215', prec=30), Mul(Integer(-1), Float('1.19182294662742561401495340245672', prec=30), I)), Integer(-1)), Mul(Add(Float('0.061686779567832924167854223579848', prec=30), Mul(Integer(-1), Float('1.90504097930308129535280142244488', prec=30), I)), Symbol('n0', commutative=False))), Mul(Pow(Add(Symbol('t', real=True), Float('-0.3678453861815398380437964998596', prec=30), Mul(Integer(-1), Float('3.65812129867866730352792994532021', prec=30), I)), Integer(-1)), Mul(Add(Mul(Integer(-1), Float('0.073395957163942207458442844011801', prec=30)), Mul(Float('0.449999922474062719432845030483482', prec=30), I)), Symbol('n0', commutative=False)))))))""",
    locals=globals())
    customre # silence pyflakes

    lrat_func = lambdify((t, n0), rat_func4*n0, 'numpy')
    lpart_frac = lambdify((t, n0), part_frac4*n0, 'numpy')
    lpart_frac_complex = lambdify((t, n0), part_frac_complex4*n0, ['numpy', {'customre': np.real}])

    lrat_func_vec = lambdify((t, n0), rat_func_vec, 'numpy')
    lrat_func_horner_vec = lambdify((t, n0), rat_func_horner_vec, 'numpy')
    lpart_frac_vec = lambdify((t, n0), part_frac_vec, 'numpy')
    lpart_frac_complex_vec = lambdify((t, n0), part_frac_complex_vec, ['numpy', {'customre': np.real}])

    tvals = np.random.random(10)
    n0vals = np.random.random(10)

    np.testing.assert_allclose(lrat_func(tvals, n0vals), lrat_func_vec(tvals, n0vals))
    np.testing.assert_allclose(lrat_func(tvals, n0vals), lrat_func_horner_vec(tvals, n0vals))
    np.testing.assert_allclose(lpart_frac(tvals, n0vals), lpart_frac_vec(tvals, n0vals))
    np.testing.assert_allclose(lpart_frac_complex(tvals, n0vals), lpart_frac_complex_vec(tvals, n0vals))
